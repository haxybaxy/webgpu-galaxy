#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
public:
    Camera();

    void setTarget(const glm::vec3 &target) { target_ = target; }
    void setDistance(float distance) { distance_ = distance; }

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix(float aspect) const;
    glm::vec3 getPosition() const;

    void onMouseMove(double x, double y);
    void onMouseButton(int button, int action, int mods);
    void onScroll(double yoffset);
    void update(float dt, bool keyW, bool keyA, bool keyS, bool keyD,
                bool keySpace, bool keyShift);

private:
    void clampPitch();
    glm::vec3 getForward() const;
    glm::vec3 getRight() const;

    glm::vec3 target_{0.0f, 0.0f, 0.0f};
    float distance_ = 80.0f;
    float yaw_ = 0.0f;
    float pitch_ = 0.3f;
    float fovY_ = glm::radians(45.0f);
    float nearZ_ = 0.1f;
    float farZ_ = 10000.0f;

    float lookSensitivity_ = 0.005f;
    float zoomSensitivity_ = 5.0f;
    float panSpeed_ = 30.0f;

    bool dragging_ = false;
    double lastMouseX_ = 0.0;
    double lastMouseY_ = 0.0;
};
