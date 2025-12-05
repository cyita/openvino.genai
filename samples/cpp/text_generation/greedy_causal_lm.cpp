// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (3 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\"");

    std::string models_path = argv[1];
    std::string prompt = argv[2];
    std::string device = "GPU";  // GPU can be used as well
    ov::AnyMap pipe_config = {};
    pipe_config["enable_save_ov_model"] = true;
    pipe_config.insert({ov::cache_dir("llm_cache")});

    ov::genai::LLMPipeline pipe(models_path, device, pipe_config);
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    size_t num_warmup = 1;
    size_t num_iter = 3;

    auto streamer = [](std::string word) {
        std::cout << word << std::flush;
        // Return flag corresponds whether generation should be stopped.
        return ov::genai::StreamingStatus::RUNNING;
    };

    // std::cout << "question:\n";
    // std::getline(std::cin, prompt);
    pipe.generate(prompt, config, streamer);
    std::cout << "\n----------\n"
                 "generation finished\n";

    std::cout << "start benchmarking: once warmup and run 3 interations\n";

    for (size_t i = 0; i < num_warmup; i++)
        pipe.generate(prompt, config);

    ov::genai::DecodedResults res = pipe.generate(prompt, config);
    ov::genai::PerfMetrics metrics = res.perf_metrics;

    for (size_t i = 0; i < num_iter - 1; i++) {
        res = pipe.generate(prompt, config);
        metrics = metrics + res.perf_metrics;
    }
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Load time: " << metrics.get_load_time() << " ms" << std::endl;
    std::cout << "Generate time: " << metrics.get_generate_duration().mean << " ± " << metrics.get_generate_duration().std << " ms" << std::endl;
    std::cout << "Tokenization time: " << metrics.get_tokenization_duration().mean << " ± " << metrics.get_tokenization_duration().std << " ms" << std::endl;
    std::cout << "Detokenization time: " << metrics.get_detokenization_duration().mean << " ± " << metrics.get_detokenization_duration().std << " ms" << std::endl;
    std::cout << "TTFT: " << metrics.get_ttft().mean  << " ± " << metrics.get_ttft().std << " ms" << std::endl;
    std::cout << "TPOT: " << metrics.get_tpot().mean  << " ± " << metrics.get_tpot().std << " ms/token " << std::endl;
    std::cout << "Throughput: " << metrics.get_throughput().mean  << " ± " << metrics.get_throughput().std << " tokens/s" << std::endl;

} catch (const std::exception& error) {

    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}

    return EXIT_FAILURE;

} catch (...) {

    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}

    return EXIT_FAILURE;
}
