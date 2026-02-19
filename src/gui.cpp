#include "gui.hpp"
#include "wgpu_utils.hpp"
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_wgpu.h>
#include <imgui.h>

using namespace wgpu_utils;

GUI::GUI() {}
GUI::~GUI() { shutdown(); }

void GUI::initialize(WGPUDevice device, WGPUTextureFormat surfaceFormat,
                     graphics::Window *window) {
    device_ = device;
    surfaceFormat_ = surfaceFormat;
    window_ = window;
    initImGui();
}

void GUI::initImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOther(window_, true);
    ImGui_ImplWGPU_InitInfo initInfo{};
    initInfo.Device = device_;
    initInfo.NumFramesInFlight = 3;
    initInfo.RenderTargetFormat = surfaceFormat_;
    initInfo.DepthStencilFormat = WGPUTextureFormat_Undefined;
    initInfo.PipelineMultisampleState.count = 1;
    ImGui_ImplWGPU_Init(&initInfo);
    imguiReady_ = true;
}

void GUI::beginFrame(float deltaTime) {
    if (!imguiReady_) return;
    (void)deltaTime;

    ImGui_ImplWGPU_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowSize(ImVec2(300, 350), ImGuiCond_FirstUseEver);
    ImGui::Begin("Simulation Controls");

    ImGui::Text("FPS: %.1f", fps_);
    ImGui::Text("Particles: %d", particleCount_);
    ImGui::Text("Octree Nodes: %d", nodeCount_);
    ImGui::Separator();

    if (params_.paused) {
        if (ImGui::Button("Play")) {
            params_.paused = false;
        }
    } else {
        if (ImGui::Button("Pause")) {
            params_.paused = true;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Step")) {
        params_.stepOnce = true;
    }
    ImGui::Separator();

    ImGui::SliderFloat("Timestep", &params_.dt, 0.0001f, 0.01f, "%.4f");
    ImGui::SliderFloat("Softening", &params_.softening, 0.01f, 5.0f, "%.2f");
    ImGui::SliderFloat("Theta (BH)", &params_.theta, 0.1f, 2.0f, "%.2f");
    ImGui::TextWrapped("Lower theta = more accurate but slower");

    ImGui::Separator();
    ImGui::InputInt("Particle Count", &params_.numParticles, 1000, 5000);
    if (params_.numParticles < 100) params_.numParticles = 100;
    if (params_.numParticles > 100000) params_.numParticles = 100000;

    ImGui::Separator();
    ImGui::Checkbox("Show Octree (debug)", &params_.showOctree);

    ImGui::End();
    ImGui::Render();
}

void GUI::render(WGPUTextureView targetView) {
    if (!imguiReady_) return;

    WGPUCommandEncoder encoder = createCommandEncoder(device_);
    WGPURenderPassColorAttachment colorAttachment{};
    colorAttachment.view = targetView;
    colorAttachment.resolveTarget = nullptr;
    colorAttachment.loadOp = WGPULoadOp_Load;
    colorAttachment.storeOp = WGPUStoreOp_Store;
    colorAttachment.clearValue = {0, 0, 0, 1};
#ifndef WEBGPU_BACKEND_WGPU
    colorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
#endif
    WGPURenderPassDescriptor passDesc{};
    passDesc.colorAttachmentCount = 1;
    passDesc.colorAttachments = &colorAttachment;
    passDesc.depthStencilAttachment = nullptr;
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc);
    ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
    WGPUQueue queue = wgpuDeviceGetQueue(device_);
    finishCommandEncoder(queue, encoder);
    wgpuQueueRelease(queue);
}

void GUI::shutdown() { shutdownImGui(); }

void GUI::shutdownImGui() {
    if (!imguiReady_) return;
    ImGui_ImplWGPU_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    imguiReady_ = false;
}
