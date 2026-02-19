#include "octree.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

int Octree::allocNode(const glm::vec3 &center, float halfWidth) {
    BuildNode node;
    node.center = center;
    node.halfWidth = halfWidth;
    buildNodes_.push_back(node);
    return static_cast<int>(buildNodes_.size() - 1);
}

int Octree::getOctant(const glm::vec3 &center, const glm::vec3 &point) const {
    int octant = 0;
    if (point.x >= center.x) octant |= 1;
    if (point.y >= center.y) octant |= 2;
    if (point.z >= center.z) octant |= 4;
    return octant;
}

glm::vec3 Octree::childCenter(const glm::vec3 &parentCenter, float parentHalfWidth, int octant) const {
    float quarter = parentHalfWidth * 0.5f;
    return glm::vec3(
        parentCenter.x + ((octant & 1) ? quarter : -quarter),
        parentCenter.y + ((octant & 2) ? quarter : -quarter),
        parentCenter.z + ((octant & 4) ? quarter : -quarter)
    );
}

void Octree::insert(int nodeIdx, int particleIdx, const glm::vec4 &pos) {
    BuildNode &node = buildNodes_[nodeIdx];
    glm::vec3 point(pos.x, pos.y, pos.z);
    float mass = pos.w;

    if (node.isLeaf && node.particleCount == 0) {
        node.particleIndex = particleIdx;
        node.particleCount = 1;
        node.comSum = point * mass;
        node.massSum = mass;
        return;
    }

    if (node.isLeaf && node.particleCount == 1) {
        node.isLeaf = false;
        int existingIdx = node.particleIndex;
        node.particleIndex = -1;

        glm::vec3 existingPos(positions_[existingIdx]);
        int existingOctant = getOctant(node.center, existingPos);
        float childHalf = node.halfWidth * 0.5f;

        if (node.childIndices[existingOctant] < 0) {
            glm::vec3 cc = childCenter(node.center, node.halfWidth, existingOctant);
            node.childIndices[existingOctant] = allocNode(cc, childHalf);
        }
        insert(node.childIndices[existingOctant], existingIdx, positions_[existingIdx]);
    }

    // Now insert the new particle
    node.comSum += point * mass;
    node.massSum += mass;
    node.particleCount++;

    int octant = getOctant(node.center, point);
    float childHalf = node.halfWidth * 0.5f;

    // Re-read node reference after potential reallocation
    BuildNode &nodeRef = buildNodes_[nodeIdx];
    if (nodeRef.childIndices[octant] < 0) {
        glm::vec3 cc = childCenter(nodeRef.center, nodeRef.halfWidth, octant);
        nodeRef.childIndices[octant] = allocNode(cc, childHalf);
    }
    insert(buildNodes_[nodeIdx].childIndices[octant], particleIdx, pos);
}

void Octree::computeCOM(int nodeIdx) {
    BuildNode &node = buildNodes_[nodeIdx];

    if (node.isLeaf) {
        return;
    }

    node.comSum = glm::vec3(0.0f);
    node.massSum = 0.0f;

    for (int i = 0; i < 8; i++) {
        if (node.childIndices[i] >= 0) {
            computeCOM(node.childIndices[i]);
            BuildNode &child = buildNodes_[node.childIndices[i]];
            node.comSum += child.comSum;
            node.massSum += child.massSum;
        }
    }
}

void Octree::flatten() {
    nodes_.clear();
    nodes_.resize(buildNodes_.size());

    for (size_t i = 0; i < buildNodes_.size(); i++) {
        const BuildNode &bn = buildNodes_[i];
        OctreeNode &on = nodes_[i];

        glm::vec3 com = (bn.massSum > 0.0f) ? (bn.comSum / bn.massSum) : bn.center;
        on.centerOfMass = glm::vec4(com, bn.massSum);
        on.bounds = glm::vec4(bn.center, bn.halfWidth);

        for (int c = 0; c < 8; c++) {
            on.children[c] = bn.childIndices[c];
        }
    }
}

void Octree::build(const std::vector<glm::vec4> &positions) {
    buildNodes_.clear();
    nodes_.clear();
    positions_ = positions;

    if (positions.empty()) return;

    // Compute bounding box
    glm::vec3 minBound(std::numeric_limits<float>::max());
    glm::vec3 maxBound(std::numeric_limits<float>::lowest());

    for (const auto &p : positions) {
        glm::vec3 pos(p);
        minBound = glm::min(minBound, pos);
        maxBound = glm::max(maxBound, pos);
    }

    glm::vec3 center = (minBound + maxBound) * 0.5f;
    glm::vec3 extent = maxBound - minBound;
    float halfWidth = std::max({extent.x, extent.y, extent.z}) * 0.5f + 1.0f;

    // Reserve to avoid excessive reallocation
    buildNodes_.reserve(positions.size() * 2);

    allocNode(center, halfWidth);

    for (size_t i = 0; i < positions.size(); i++) {
        insert(0, static_cast<int>(i), positions[i]);
    }

    computeCOM(0);
    flatten();
}
