#include "camera.hpp"
#include <cmath>
#include <algorithm>
#include <glm/gtc/constants.hpp>

namespace {
constexpr float kPitchLimit = glm::radians(89.0f);
}

Camera::Camera() {}

glm::vec3 Camera::getPosition() const {
    float x = target_.x + distance_ * std::cos(pitch_) * std::cos(yaw_);
    float y = target_.y + distance_ * std::sin(pitch_);
    float z = target_.z + distance_ * std::cos(pitch_) * std::sin(yaw_);
    return glm::vec3(x, y, z);
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(getPosition(), target_, glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 Camera::getProjectionMatrix(float aspect) const {
    return glm::perspective(fovY_, aspect, nearZ_, farZ_);
}

glm::vec3 Camera::getForward() const {
    return glm::normalize(target_ - getPosition());
}

glm::vec3 Camera::getRight() const {
    return glm::normalize(glm::cross(getForward(), glm::vec3(0.0f, 1.0f, 0.0f)));
}

void Camera::onMouseButton(int button, int action, int /*mods*/) {
    if (button == 0) { // Left mouse button
        dragging_ = (action == 1); // 1 = press
        if (!dragging_) {
            lastMouseX_ = 0.0;
            lastMouseY_ = 0.0;
        }
    }
}

void Camera::onMouseMove(double x, double y) {
    if (!dragging_) {
        lastMouseX_ = x;
        lastMouseY_ = y;
        return;
    }

    if (lastMouseX_ == 0.0 && lastMouseY_ == 0.0) {
        lastMouseX_ = x;
        lastMouseY_ = y;
        return;
    }

    double dx = x - lastMouseX_;
    double dy = y - lastMouseY_;
    lastMouseX_ = x;
    lastMouseY_ = y;

    yaw_ += static_cast<float>(dx) * lookSensitivity_;
    pitch_ -= static_cast<float>(dy) * lookSensitivity_;
    clampPitch();
}

void Camera::onScroll(double yoffset) {
    distance_ -= static_cast<float>(yoffset) * zoomSensitivity_;
    distance_ = std::max(1.0f, distance_);
}

void Camera::update(float dt, bool keyW, bool keyA, bool keyS, bool keyD,
                    bool keySpace, bool keyShift) {
    glm::vec3 forward = getForward();
    glm::vec3 right = getRight();
    // Project forward onto XZ plane for panning
    glm::vec3 forwardXZ = glm::normalize(glm::vec3(forward.x, 0.0f, forward.z));

    glm::vec3 move(0.0f);
    if (keyW) move += forwardXZ;
    if (keyS) move -= forwardXZ;
    if (keyD) move += right;
    if (keyA) move -= right;
    if (keySpace) move.y += 1.0f;
    if (keyShift) move.y -= 1.0f;

    if (glm::dot(move, move) > 0.0f) {
        move = glm::normalize(move);
        target_ += move * panSpeed_ * dt;
    }
}

void Camera::clampPitch() {
    pitch_ = glm::clamp(pitch_, -kPitchLimit, kPitchLimit);
}
