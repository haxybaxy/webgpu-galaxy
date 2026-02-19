#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

struct OctreeNode {
    glm::vec4 centerOfMass;  // xyz = center of mass, w = total mass
    glm::vec4 bounds;        // xyz = center of bounding box, w = half-width
    int32_t children[8];     // Child indices (-1 = empty)
};

class Octree {
public:
    void build(const std::vector<glm::vec4> &positions);
    const std::vector<OctreeNode> &getNodes() const { return nodes_; }
    size_t nodeCount() const { return nodes_.size(); }

private:
    struct BuildNode {
        glm::vec3 center;
        float halfWidth;
        glm::vec3 comSum{0.0f};
        float massSum = 0.0f;
        int particleIndex = -1;
        int childIndices[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
        bool isLeaf = true;
        int particleCount = 0;
    };

    int allocNode(const glm::vec3 &center, float halfWidth);
    void insert(int nodeIdx, int particleIdx, const glm::vec4 &pos);
    int getOctant(const glm::vec3 &center, const glm::vec3 &point) const;
    glm::vec3 childCenter(const glm::vec3 &parentCenter, float parentHalfWidth, int octant) const;
    void computeCOM(int nodeIdx);
    void flatten();

    std::vector<BuildNode> buildNodes_;
    std::vector<glm::vec4> positions_;
    std::vector<OctreeNode> nodes_;
};
