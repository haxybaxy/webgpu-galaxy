#pragma once

#include <tuple>
#include <webgpu/webgpu.h>

namespace wgpu_utils {

WGPUInstance requestInstance();

WGPUAdapter requestAdapterSync(WGPUInstance instance,
                               WGPURequestAdapterOptions const *options);

WGPUDevice requestDeviceSync(WGPUAdapter adapter,
                             WGPUDeviceDescriptor const *descriptor);

void printAdapterInfo(WGPUAdapter adapter);
void printDeviceInfo(WGPUDevice device);

std::tuple<WGPUInstance, WGPUDevice, WGPUAdapter> initializeWebGPU();

void deviceWait(WGPUDevice device, const bool *isDone);

void wgpuPollEvents([[maybe_unused]] WGPUDevice device,
                    [[maybe_unused]] bool yieldToWebBrowser);

class Buffer {
    WGPUBuffer m_buffer;
    WGPUDevice m_device;
    WGPUBufferUsageFlags m_usage;
    size_t m_size;
    size_t m_capacity;

public:
    Buffer();
    void initialize(WGPUDevice device, WGPUBufferUsageFlags usage, size_t size = 0);
    ~Buffer() noexcept;

    WGPUBuffer get() const;
    size_t size() const { return m_size; }
    size_t capacity() const { return m_capacity; }
    void clear();

    void upload(void *begin, size_t bytes, bool waitForCompletion = false);
    void upload(WGPUQueue queue, void *begin, size_t bytes, bool waitForCompletion = false);
};

WGPUShaderModule createShaderModule(WGPUDevice device, const char *shaderSource);

WGPUCommandEncoder createCommandEncoder(WGPUDevice device,
                                        WGPUCommandEncoderDescriptor *desc = nullptr);

void finishCommandEncoder(WGPUQueue queue, WGPUCommandEncoder encoder);

const void *mapBufferSync(WGPUDevice device, WGPUBuffer buffer, size_t size);

} // namespace wgpu_utils
