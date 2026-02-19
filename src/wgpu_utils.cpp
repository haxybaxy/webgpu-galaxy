#include "wgpu_utils.hpp"
#include <vector>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif
#ifdef WEBGPU_BACKEND_WGPU
#include <webgpu/wgpu.h>
#endif
#ifdef WEBGPU_BACKEND_DAWN
#include <dawn/webgpu.h>
#endif
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace wgpu_utils {

WGPUInstance requestInstance() {
    spdlog::debug("Creating WGPU instance...");
    WGPUInstanceDescriptor desc = {};
#ifdef WEBGPU_BACKEND_DAWN
    static const char *kEnabledToggles[] = {"allow_unsafe_apis"};
    WGPUDawnTogglesDescriptor dawnToggles{};
    dawnToggles.chain.sType = WGPUSType_DawnTogglesDescriptor;
    dawnToggles.chain.next = nullptr;
    dawnToggles.enabledToggles = kEnabledToggles;
    dawnToggles.enabledTogglesCount = 1;
    dawnToggles.disabledToggles = nullptr;
    dawnToggles.disabledTogglesCount = 0;
    desc.nextInChain = reinterpret_cast<const WGPUChainedStruct *>(&dawnToggles);
#else
    desc.nextInChain = nullptr;
#endif
#ifdef WEBGPU_BACKEND_EMSCRIPTEN
    WGPUInstance instance = wgpuCreateInstance(nullptr);
#else
    WGPUInstance instance = wgpuCreateInstance(&desc);
#endif
    if (!instance) {
        throw std::runtime_error("Failed to create WGPU instance");
    }
    spdlog::info("WGPU instance created: {:#x}", size_t(instance));
    return instance;
}

WGPUAdapter requestAdapterSync(WGPUInstance instance,
                               WGPURequestAdapterOptions const *options) {
    struct UserData {
        WGPUAdapter adapter = nullptr;
        bool requestEnded = false;
    };
    UserData userData;
    auto onAdapterRequestEnded = [](WGPURequestAdapterStatus status,
                                    WGPUAdapter adapter, char const *message,
                                    void *pUserData) {
        UserData &userData = *reinterpret_cast<UserData *>(pUserData);
        if (status == WGPURequestAdapterStatus_Success) {
            userData.adapter = adapter;
        } else {
            spdlog::error("Adapter request failed:");
            if (message) spdlog::error("{}", message);
        }
        userData.requestEnded = true;
    };
    wgpuInstanceRequestAdapter(instance, options, onAdapterRequestEnded, (void *)&userData);
#ifdef __EMSCRIPTEN__
    while (!userData.requestEnded) {
        emscripten_sleep(10);
    }
    if (!userData.requestEnded || !userData.adapter) {
        spdlog::error("Adapter request did not complete or failed");
        emscripten_force_exit(1);
    }
#else
    if (!userData.requestEnded || !userData.adapter) {
        throw std::runtime_error("Adapter request did not complete");
    }
#endif
    return userData.adapter;
}

WGPUDevice requestDeviceSync(WGPUAdapter adapter,
                             WGPUDeviceDescriptor const *descriptor) {
    struct UserData {
        WGPUDevice device = nullptr;
        bool requestEnded = false;
    };
    UserData userData;
    auto onDeviceRequestEnded = [](WGPURequestDeviceStatus status,
                                   WGPUDevice device, char const *message,
                                   void *pUserData) {
        UserData &userData = *reinterpret_cast<UserData *>(pUserData);
        if (status == WGPURequestDeviceStatus_Success) {
            userData.device = device;
        } else {
            spdlog::error("Device request failed:");
            if (message) spdlog::error("{}", message);
        }
        userData.requestEnded = true;
    };
    wgpuAdapterRequestDevice(adapter, descriptor, onDeviceRequestEnded, (void *)&userData);
#ifdef __EMSCRIPTEN__
    while (!userData.requestEnded) {
        emscripten_sleep(100);
    }
#endif
    if (!userData.requestEnded || !userData.device) {
        throw std::runtime_error("Device request did not complete");
    }
    auto onDeviceError = [](WGPUErrorType type, char const *message, void *) {
        spdlog::error("Uncaptured device error (type {}): {}", int(type),
                      message ? message : "no message");
    };
    wgpuDeviceSetUncapturedErrorCallback(userData.device, onDeviceError, nullptr);
    return userData.device;
}

std::tuple<WGPUInstance, WGPUDevice, WGPUAdapter> initializeWebGPU() {
    WGPUInstance instance = requestInstance();
    WGPURequestAdapterOptions adapterOpts = {};
    adapterOpts.nextInChain = nullptr;
    WGPUAdapter adapter = requestAdapterSync(instance, &adapterOpts);
    spdlog::info("Got adapter: {:#x}", size_t(adapter));
    printAdapterInfo(adapter);

    WGPUDeviceDescriptor deviceDesc = {};
    deviceDesc.nextInChain = nullptr;
    deviceDesc.label = "GalaxySim Device";
    deviceDesc.requiredFeatureCount = 0;
    deviceDesc.requiredFeatures = nullptr;
    deviceDesc.requiredLimits = nullptr;
    deviceDesc.defaultQueue.nextInChain = nullptr;
    deviceDesc.defaultQueue.label = "Default queue";
    deviceDesc.deviceLostCallback = [](WGPUDeviceLostReason reason,
                                       char const *message, void *) {
        spdlog::error("Device lost (reason {}): {}", int(reason),
                      message ? message : "no message");
    };
    WGPUDevice device = requestDeviceSync(adapter, &deviceDesc);
    spdlog::info("Got device: {:#x}", size_t(device));
    printDeviceInfo(device);
    return std::make_tuple(instance, device, adapter);
}

void printAdapterInfo(WGPUAdapter adapter) {
#ifndef __EMSCRIPTEN__
    WGPUSupportedLimits supportedLimits{};
    supportedLimits.nextInChain = nullptr;
#ifdef WEBGPU_BACKEND_DAWN
    bool success = wgpuAdapterGetLimits(adapter, &supportedLimits) == WGPUStatus_Success;
#else
    bool success = wgpuAdapterGetLimits(adapter, &supportedLimits);
#endif
    if (success) {
        spdlog::info("Adapter limits:");
        spdlog::info(" - maxTextureDimension2D: {}", supportedLimits.limits.maxTextureDimension2D);
        spdlog::info(" - maxStorageBufferBindingSize: {}", supportedLimits.limits.maxStorageBufferBindingSize);
    }
#endif

    WGPUAdapterProperties properties{};
    properties.nextInChain = nullptr;
    wgpuAdapterGetProperties(adapter, &properties);
    spdlog::info("Adapter: {}", properties.name ? properties.name : "unknown");
    spdlog::info(" - backend: {:#x}", static_cast<uint32_t>(properties.backendType));
}

void printDeviceInfo(WGPUDevice device) {
    WGPUSupportedLimits limits{};
    limits.nextInChain = nullptr;
#ifdef WEBGPU_BACKEND_DAWN
    bool success = wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success;
#else
    bool success = wgpuDeviceGetLimits(device, &limits);
#endif
    if (success) {
        spdlog::info("Device limits:");
        spdlog::info(" - maxStorageBufferBindingSize: {}", limits.limits.maxStorageBufferBindingSize);
        spdlog::info(" - maxComputeWorkgroupSizeX: {}", limits.limits.maxComputeWorkgroupSizeX);
    }
}

void deviceWait(WGPUDevice device, const bool *isDone) {
    while (!*isDone) {
        wgpuPollEvents(device, true);
    }
}

void wgpuPollEvents([[maybe_unused]] WGPUDevice device,
                    [[maybe_unused]] bool yieldToWebBrowser) {
#if defined(WEBGPU_BACKEND_DAWN)
    wgpuDeviceTick(device);
#elif defined(WEBGPU_BACKEND_WGPU)
    wgpuDevicePoll(device, false, nullptr);
#elif defined(WEBGPU_BACKEND_EMSCRIPTEN)
    if (yieldToWebBrowser) {
        emscripten_sleep(0);
    }
#endif
}

Buffer::Buffer()
    : m_buffer(nullptr), m_device(nullptr), m_usage(0), m_size(0), m_capacity(0) {}

void Buffer::initialize(WGPUDevice device, WGPUBufferUsageFlags usage, size_t size) {
    if (m_buffer) {
        clear();
    }
    WGPUBufferDescriptor bufferDesc{};
    bufferDesc.nextInChain = nullptr;
    bufferDesc.size = size;
    bufferDesc.usage = usage;
    bufferDesc.mappedAtCreation = false;
    m_buffer = wgpuDeviceCreateBuffer(device, &bufferDesc);
    m_device = device;
    m_usage = usage;
    m_size = size;
    m_capacity = size;
}

Buffer::~Buffer() noexcept {
    try { clear(); } catch (...) {}
}

WGPUBuffer Buffer::get() const { return m_buffer; }

void Buffer::clear() {
    if (m_buffer) {
        wgpuBufferRelease(m_buffer);
        m_buffer = nullptr;
        m_size = 0;
        m_capacity = 0;
    }
}

void Buffer::upload(void *begin, size_t bytes, bool waitForCompletion) {
    if (!m_buffer) throw std::runtime_error("Buffer not initialized");
    WGPUQueue queue = wgpuDeviceGetQueue(m_device);
    upload(queue, begin, bytes, waitForCompletion);
    wgpuQueueRelease(queue);
}

void Buffer::upload(WGPUQueue queue, void *begin, size_t bytes, bool waitForCompletion) {
    if (!m_buffer) throw std::runtime_error("Buffer not initialized");
    if (m_capacity >= bytes) {
        m_size = bytes;
    } else {
        clear();
        initialize(m_device, m_usage, bytes);
        m_size = bytes;
        m_capacity = bytes;
    }
    wgpuQueueWriteBuffer(queue, m_buffer, 0, begin, bytes);
    wgpuPollEvents(m_device, false);
    if (waitForCompletion) {
        bool uploadDone = false;
        wgpuQueueOnSubmittedWorkDone(
            queue,
            [](WGPUQueueWorkDoneStatus, void *ud) { *static_cast<bool *>(ud) = true; },
            &uploadDone);
        while (!uploadDone) {
            wgpuPollEvents(m_device, false);
        }
    }
}

WGPUShaderModule createShaderModule(WGPUDevice device, const char *shaderSource) {
    WGPUShaderModuleDescriptor shaderDesc = {};
#ifdef WEBGPU_BACKEND_WGPU
    shaderDesc.hintCount = 0;
    shaderDesc.hints = nullptr;
#endif
    WGPUShaderModuleWGSLDescriptor shaderCodeDesc{};
    shaderCodeDesc.chain.next = nullptr;
    shaderCodeDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    shaderDesc.nextInChain = &shaderCodeDesc.chain;
    shaderCodeDesc.code = shaderSource;
    WGPUShaderModule shaderModule = wgpuDeviceCreateShaderModule(device, &shaderDesc);
    if (!shaderModule) {
        throw std::runtime_error("Failed to create shader module");
    }
    return shaderModule;
}

WGPUCommandEncoder createCommandEncoder(WGPUDevice device,
                                        WGPUCommandEncoderDescriptor *desc) {
    WGPUCommandEncoderDescriptor encoderDesc = {};
    if (desc == nullptr) {
        encoderDesc.nextInChain = nullptr;
        encoderDesc.label = "Command encoder";
        desc = &encoderDesc;
    }
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, desc);
    if (!encoder) {
        throw std::runtime_error("Failed to create command encoder");
    }
    return encoder;
}

void finishCommandEncoder(WGPUQueue queue, WGPUCommandEncoder encoder) {
    WGPUCommandBufferDescriptor desc = {};
    desc.nextInChain = nullptr;
    desc.label = "Command buffer";
    WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, &desc);
    wgpuCommandEncoderRelease(encoder);
    if (!command) {
        throw std::runtime_error("Failed to finish command encoder");
    }
    wgpuQueueSubmit(queue, 1, &command);
    wgpuCommandBufferRelease(command);
}

const void *mapBufferSync(WGPUDevice device, WGPUBuffer buffer, size_t size) {
    struct MapContext {
        bool done = false;
        const void *data = nullptr;
        WGPUBuffer buf;
        size_t sz;
    } context{false, nullptr, buffer, size};

    wgpuBufferMapAsync(
        buffer, WGPUMapMode_Read, 0, size,
        [](WGPUBufferMapAsyncStatus status, void *userData) {
            auto *ctx = static_cast<MapContext *>(userData);
            if (status == WGPUBufferMapAsyncStatus_Success) {
                ctx->data = wgpuBufferGetConstMappedRange(ctx->buf, 0, ctx->sz);
            }
            ctx->done = true;
        },
        &context);

    while (!context.done) {
        wgpuPollEvents(device, true);
    }
    return context.data;
}

} // namespace wgpu_utils
