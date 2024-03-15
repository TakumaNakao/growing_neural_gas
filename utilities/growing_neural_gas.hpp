#pragma once

#include <memory>
#include <tuple>
#include <limits>

#include <Eigen/Dense>
#include <matplot/matplot.h>

class GrowingNeuralGas {
public:
    class Edge;
    class Node {
    public:
        using SharedPtr = std::shared_ptr<Node>;

    private:
        Eigen::Vector3d p_;
        std::vector<std::weak_ptr<Edge>> edges_;
        double error_ = 0;

    public:
        Node(Eigen::Vector3d p, std::vector<std::weak_ptr<Edge>> edges = {}) : p_(p), edges_(edges) {}
        void add_edge(std::weak_ptr<Edge> edge) { edges_.push_back(edge); }
        void update_p(const Eigen::Vector3d& p, double eps) { p_ += eps * (p - p_); }
        void set_error(double error) { error_ = error; }
        Eigen::Vector3d p() const { return p_; }
        double error() const { return error_; }
        std::vector<std::weak_ptr<Edge>> edges() const { return edges_; }
    };
    class Edge {
    public:
        using SharedPtr = std::shared_ptr<Edge>;

    private:
        std::array<Node::SharedPtr, 2> node_ptrs_;
        double distance_;

    public:
        size_t age;
        Edge(std::array<Node::SharedPtr, 2> node_ptrs, size_t _age = 0) : node_ptrs_(node_ptrs), age(_age)
        {
            assert(node_ptrs_[0] && node_ptrs_[1]);
            distance_ = (node_ptrs_[0]->p() - node_ptrs_[1]->p()).norm();
        }
        Edge(Node::SharedPtr node_ptr1, Node::SharedPtr node_ptr2, size_t _age = 0) : Edge({node_ptr1, node_ptr2}, _age) {}
        Node::SharedPtr get_connect_node(Node::SharedPtr node) const
        {
            assert(node_ptrs_[0] == node || node_ptrs_[1] == node);
            return node_ptrs_[0] == node ? node_ptrs_[1] : node_ptrs_[0];
        }
        double distance() const { return distance_; }
        std::array<Node::SharedPtr, 2> node_ptrs() const { return node_ptrs_; }
    };

private:
    std::vector<Node::SharedPtr> nodes_;
    std::vector<Edge::SharedPtr> edges_;
    size_t update_count_ = 0;

    std::tuple<Node::SharedPtr, Node::SharedPtr> serch_shortest_node(const Eigen::Vector3d& p) const
    {
        assert(nodes_.size() >= 2);
        double first_min_distance = std::numeric_limits<double>::max();
        double second_min_distance = std::numeric_limits<double>::max();
        Node::SharedPtr first_min_node = nullptr;
        Node::SharedPtr seconsd_min_node = nullptr;
        for (const auto& node : nodes_) {
            if (double d = (node->p() - p).norm(); d < first_min_distance) {
                second_min_distance = first_min_distance;
                seconsd_min_node = first_min_node;
                first_min_distance = d;
                first_min_node = node;
            }
            else if (d < second_min_distance) {
                second_min_distance = d;
                seconsd_min_node = node;
            }
        }
        return {first_min_node, seconsd_min_node};
    }

    void add_edge(Node::SharedPtr node_ptr1, Node::SharedPtr node_ptr2)
    {
        edges_.push_back(std::make_shared<Edge>(node_ptr1, node_ptr2));
        node_ptr1->add_edge(edges_.back());
        node_ptr2->add_edge(edges_.back());
    }

public:
    GrowingNeuralGas(std::array<Eigen::Vector3d, 2> points)
    {
        for (const auto& p : points) {
            nodes_.push_back(std::make_shared<Node>(p));
        }
        add_edge(nodes_[0], nodes_[1]);
    }
    void update(const Eigen::Vector3d& p, double eps1, double eps2, double max_age, size_t lambda, double alpha, double beta)
    {
        {
            auto [first_min_node, second_min_node] = serch_shortest_node(p);
            first_min_node->set_error(first_min_node->error() + (first_min_node->p() - p).norm());
            first_min_node->update_p(p, eps1);
            bool is_connected_first_second = false;
            for (const auto& edge : first_min_node->edges()) {
                if (auto r_edge = edge.lock()) {
                    auto r_node = r_edge->get_connect_node(first_min_node);
                    r_node->update_p(p, eps2);
                    if (r_node == second_min_node) {
                        r_edge->age = 0;
                        is_connected_first_second = true;
                    }
                    else {
                        r_edge->age++;
                    }
                }
            }
            if (!is_connected_first_second) {
                add_edge(first_min_node, second_min_node);
            }
        }
        {
            std::erase_if(edges_, [&max_age](Edge::SharedPtr edge) { return edge->age > max_age; });
            std::erase_if(nodes_, [](Node::SharedPtr node) { return node.use_count() == 1; });
        }

        update_count_++;

        if (update_count_ >= lambda) {
            update_count_ = 0;
            auto max_error_node = *std::max_element(nodes_.begin(), nodes_.end(), [](const Node::SharedPtr& a, const Node::SharedPtr& b) { return a->error() < b->error(); });
            double max_distance = 0;
            std::shared_ptr<Edge> max_distance_edge = nullptr;
            for (const auto& edge : max_error_node->edges()) {
                if (auto r = edge.lock()) {
                    if (max_distance < r->distance()) {
                        max_distance = r->distance();
                        max_distance_edge = r;
                    }
                }
            }
            auto max_distance_node = max_distance_edge->get_connect_node(max_error_node);
            Eigen::Vector3d new_p = 0.5 * (max_error_node->p() + max_distance_node->p());
            auto new_node = std::make_shared<Node>(new_p);
            nodes_.push_back(new_node);
            std::erase_if(edges_, [&max_distance_edge](Edge::SharedPtr edge) { return edge == max_distance_edge; });
            add_edge(new_node, max_error_node);
            add_edge(new_node, max_distance_node);

            max_error_node->set_error(alpha * max_error_node->error());
            max_distance_node->set_error(alpha * max_distance_node->error());
            new_node->set_error(0.5 * (max_error_node->error() + max_distance_node->error()));
            for (auto& node : nodes_) {
                node->set_error(beta * node->error());
            }
        }
    }
    void plot() const
    {
        // {
        //     std::vector<double> x, y, z;
        //     x.reserve(nodes_.size());
        //     y.reserve(nodes_.size());
        //     z.reserve(nodes_.size());
        //     for (const auto& node : nodes_) {
        //         auto p = node->p();
        //         x.push_back(p(0));
        //         y.push_back(p(1));
        //         z.push_back(p(2));
        //     }
        //     matplot::plot3(x, y, z, "o")->marker_size(6).marker_color("black");
        // }
        for (const auto& edge : edges_) {
            std::vector<double> x, y, z;
            for (const auto& node : edge->node_ptrs()) {
                auto p = node->p();
                x.push_back(p(0));
                y.push_back(p(1));
                z.push_back(p(2));
            }
            matplot::plot3(x, y, z, "-o")->line_width(3).color("black");
        }
    }
};