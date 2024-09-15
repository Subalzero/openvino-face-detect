#ifndef OPENVINO_RUNNER_H
#define OPENVINO_RUNNER_H

#include <iostream>
#include <string>
#include <cassert>
#include <queue>
#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable>

#include "openvino/openvino.hpp"

class OpenVINORunner
{
public:
	OpenVINORunner();
	~OpenVINORunner();

	OpenVINORunner(const std::string& xml_path, const std::string& default_device);

	OpenVINORunner(OpenVINORunner&& temp) noexcept;
	OpenVINORunner& operator=(OpenVINORunner&& temp) noexcept;

	OpenVINORunner(const OpenVINORunner& copy) = delete;
	OpenVINORunner& operator=(const OpenVINORunner& copy) = delete;

	std::vector<size_t> get_input_shape() const;
	std::vector<size_t> get_output_shape() const;

	void process(const std::vector<float>& data);
	void process_async(const std::vector<float>& data);

	void get(std::vector<float>& output);

	static std::vector<std::string> get_available_devices();
	static std::string get_device_name(const std::string& device);

private:
	static ov::Core _core;

	std::string _model_xml;
	std::string _device;

	ov::CompiledModel _compiled_model;
	ov::Shape _input_shape;
	ov::Shape _output_shape;

	std::vector<ov::InferRequest> _infer_requests;
	std::queue<ov::InferRequest*> _idle_requests;
	std::queue<ov::InferRequest*> _busy_requests;

	std::queue<std::vector<float>> _infer_results;

	std::mutex _idle_requests_mut;
	std::mutex _busy_requests_mut;
	std::mutex _infer_results_mut;

	std::condition_variable _idle_requests_cond;
	std::condition_variable _busy_requests_cond;
	std::condition_variable _infer_results_cond;
};

#endif // OPENVINO_RUNNER_H