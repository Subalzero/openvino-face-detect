#include "openvinorunner.h"

#include <memory>

OpenVINORunner::OpenVINORunner()
{
}

OpenVINORunner::~OpenVINORunner()
{
}

OpenVINORunner::OpenVINORunner(const std::string& xml_path, const std::string& default_device)
{
	_model_xml = xml_path;
	_device = default_device;

	std::shared_ptr<ov::Model> model = _core.read_model(_model_xml);
	_input_shape = model->input().get_shape();
	_output_shape = model->get_output_shape();

	_compiled_model = _core.compile_model(model, _device,
		ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
		ov::hint::inference_precision(ov::element::f32));

	uint32_t nireq = _compiled_model.get_property(ov::optimal_number_of_infer_requests);

	for (uint32_t i = 0; i < nireq; ++i)
	{
		_infer_requests.push_back(_compiled_model.create_infer_request());
	}

	for (size_t i = 0; i < _infer_requests.size(); ++i)
	{
		_idle_requests.push(&_infer_requests[i]);
	}
}

OpenVINORunner::OpenVINORunner(OpenVINORunner&& temp) noexcept
{
	_model_xml = std::move(temp._model_xml);
	_device = std::move(temp._device);

	_compiled_model = std::move(temp._compiled_model);
	_input_shape = _compiled_model.input().get_shape();
	_output_shape = _compiled_model.output().get_shape();

	_infer_requests = std::move(temp._infer_requests);
	
	_idle_requests = std::move(temp._idle_requests);
	_busy_requests = std::move(temp._busy_requests);
	_infer_results = std::move(temp._infer_results);

	_model_xml = std::string();
	_device = std::string();
}

OpenVINORunner& OpenVINORunner::operator=(OpenVINORunner&& temp) noexcept
{
	_model_xml = std::move(temp._model_xml);
	_device = std::move(temp._device);

	_compiled_model = std::move(temp._compiled_model);
	_input_shape = _compiled_model.input().get_shape();
	_output_shape = _compiled_model.output().get_shape();

	_infer_requests = std::move(temp._infer_requests);

	_idle_requests = std::move(temp._idle_requests);
	_busy_requests = std::move(temp._busy_requests);
	_infer_results = std::move(temp._infer_results);

	_model_xml = std::string();
	_device = std::string();

	return *this;
}

std::vector<size_t> OpenVINORunner::get_input_shape() const
{
	return _input_shape;
}

std::vector<size_t> OpenVINORunner::get_output_shape() const
{
	return _output_shape;
}

void OpenVINORunner::process(const std::vector<float>& data)
{
	assert(_model_xml.size() > 0);

	std::unique_lock<std::mutex> idle_requests_lock(_idle_requests_mut);
	_idle_requests_cond.wait(idle_requests_lock, [&]() {  return !_idle_requests.empty();  });
	ov::InferRequest* request = _idle_requests.front();
	_idle_requests.pop();
	idle_requests_lock.unlock();

	float* raw_data = const_cast<float*>(data.data());
	ov::Tensor input_tensor(_compiled_model.input().get_element_type(), _input_shape, raw_data);
	request->set_tensor(_compiled_model.input(), input_tensor);

	std::unique_lock<std::mutex> busy_requests_lock(_busy_requests_mut);
	_busy_requests.push(request);
	_busy_requests_cond.notify_one();
	busy_requests_lock.unlock();

	request->infer();

	ov::Tensor output_tensor = request->get_output_tensor();
	float* output_data = static_cast<float*>(output_tensor.data());

	std::unique_lock<std::mutex> infer_results_lock(_infer_results_mut);
	_infer_results.push(std::vector<float>(output_data, output_data + output_tensor.get_size()));
	_infer_results_cond.notify_one();
	infer_results_lock.unlock();

	busy_requests_lock.lock();
	_busy_requests_cond.wait(busy_requests_lock, [&]() { return !_busy_requests.empty(); });
	_busy_requests.pop();
	busy_requests_lock.unlock();

	idle_requests_lock.lock();
	_idle_requests.push(request);
	_idle_requests_cond.notify_one();
	idle_requests_lock.unlock();
}

void OpenVINORunner::process_async(const std::vector<float>& data)
{
	std::thread(&OpenVINORunner::process, this, data).detach();
}

void OpenVINORunner::get(std::vector<float>& output)
{
	assert(_model_xml.size() > 0);

	std::unique_lock<std::mutex> lock(_infer_results_mut);
	_infer_results_cond.wait(lock, [&]() { return !_infer_results.empty(); });
	output = _infer_results.front();
	_infer_results.pop();
}


