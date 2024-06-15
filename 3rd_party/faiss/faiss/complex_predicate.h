#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <faiss/MetricType.h>

namespace faiss {
namespace complex_predicate {

inline Buffer buffer_intersect(const Buffer& a, const Buffer& b) {
    Buffer result;
    std::set_intersection(
            a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
    return result;
}

inline Buffer buffer_union(const Buffer& a, const Buffer& b) {
    Buffer result;
    std::set_union(
            a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
    return result;
}

inline Buffer buffer_difference(const Buffer& a, const Buffer& b) {
    Buffer result;
    std::set_difference(
            a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
    return result;
}

class StateNode {
   public:
    enum class Type {
        NONE,   // No vector satisfies the predicate
        SOME,   // Some vectors (+ short list) satisfy the predicate
        MOST,   // Most vectors (- exclude list) satisfy the predicate
        ALL,    // All vectors satisfy the predicate
        UNKNOWN // Cannot determine which vectors satisfy the predicate
    };

    Type type = Type::UNKNOWN;
    Buffer short_list;
    Buffer exclude_list;

    StateNode() = default;
    StateNode(Type type, const Buffer& short_list, const Buffer& exclude_list);
    StateNode(Type type, Buffer&& short_list, Buffer&& exclude_list);

    bool operator==(const Type& otherType) const {
        return type == otherType;
    }

    bool operator==(const StateNode& other) const {
        return type == other.type && short_list == other.short_list &&
                exclude_list == other.exclude_list;
    }

    std::string type_to_str() const;

    friend std::ostream& operator<<(std::ostream& os, const StateNode& state);
};

using State = std::shared_ptr<StateNode>;
using Type = StateNode::Type;

class VarMapNode;
using VarMap = std::shared_ptr<VarMapNode>;

class VarMapNode {
   private:
    std::unordered_map<std::string, State> var_map;

   public:
    VarMapNode(std::unordered_map<std::string, State>&& var_map);

    VarMapNode(const VarMapNode& other) : var_map(other.var_map) {}

    VarMap update(const std::string& name, State new_state) const;

    VarMap update(
            const std::unordered_map<std::string, State>& new_var_map) const;

    VarMap sort() const;

    const State& get(const std::string& name) const;

    const std::unordered_map<std::string, State>& get() const {
        return var_map;
    }

    std::vector<std::string> unresolved_vars() const;

    friend std::ostream& operator<<(
            std::ostream& os,
            const VarMapNode& var_map);
};

class ExprNode {
   public:
    virtual State evaluate(VarMap var_map, bool concretize = false) const = 0;

    virtual ~ExprNode() {}

    virtual void print(std::ostream& os) const = 0;

    friend std::ostream& operator<<(std::ostream& os, const ExprNode& expr) {
        expr.print(os);
        return os;
    }
};

using Expr = std::unique_ptr<ExprNode>;

class VariableNode : public ExprNode {
   private:
    std::string var_name;

   public:
    VariableNode(const std::string& name) : var_name(name) {}

    State evaluate(VarMap var_map, bool concretize = false) const override;

    void print(std::ostream& os) const override {
        os << var_name;
    }
};

using Variable = std::unique_ptr<VariableNode>;

class UnaryOperatorNode : public ExprNode {
   protected:
    Expr operand;

   public:
    UnaryOperatorNode(Expr op) : operand(std::move(op)) {}
};

class BinaryOperatorNode : public ExprNode {
   protected:
    Expr left;
    Expr right;

   public:
    BinaryOperatorNode(Expr lhs, Expr rhs)
            : left(std::move(lhs)), right(std::move(rhs)) {}
};

class NotNode : public UnaryOperatorNode {
   public:
    using UnaryOperatorNode::UnaryOperatorNode;

    State evaluate(VarMap var_map, bool concretize = false) const override;

    void print(std::ostream& os) const override {
        os << "NOT(" << *operand << ")";
    }
};

class AndNode : public BinaryOperatorNode {
   public:
    using BinaryOperatorNode::BinaryOperatorNode;

    State evaluate(VarMap var_map, bool concretize = false) const override;

    void print(std::ostream& os) const override {
        os << "(" << *left << " AND " << *right << ")";
    }
};

class OrNode : public BinaryOperatorNode {
   public:
    using BinaryOperatorNode::BinaryOperatorNode;

    State evaluate(VarMap var_map, bool concretize = false) const override;

    void print(std::ostream& os) const override {
        os << "(" << *left << " OR " << *right << ")";
    }
};

State make_state(Type type, bool concretize = false, const Buffer& list = {});

inline const Buffer EMPTY_BUFFER = Buffer{};
inline const State STATE_NONE = std::make_shared<StateNode>(Type::NONE, EMPTY_BUFFER, EMPTY_BUFFER);
inline const State STATE_SOME = std::make_shared<StateNode>(Type::SOME, EMPTY_BUFFER, EMPTY_BUFFER);
inline const State STATE_MOST = std::make_shared<StateNode>(Type::MOST, EMPTY_BUFFER, EMPTY_BUFFER);
inline const State STATE_ALL = std::make_shared<StateNode>(Type::ALL, EMPTY_BUFFER, EMPTY_BUFFER);
inline const State STATE_UNKNOWN = std::make_shared<StateNode>(Type::UNKNOWN, EMPTY_BUFFER, EMPTY_BUFFER);

inline VarMap make_var_map(std::unordered_map<std::string, State>&& var_map) {
    return std::make_shared<VarMapNode>(std::move(var_map));
}

inline Variable make_var(const std::string& name) {
    return std::make_unique<VariableNode>(name);
}

inline Expr make_or(Expr lhs, Expr rhs) {
    return std::make_unique<OrNode>(std::move(lhs), std::move(rhs));
}

inline Expr make_and(Expr lhs, Expr rhs) {
    return std::make_unique<AndNode>(std::move(lhs), std::move(rhs));
}

inline Expr make_not(Expr operand) {
    return std::make_unique<NotNode>(std::move(operand));
}

std::vector<std::string> tokenize_formula(const std::string& formula);

Expr parse_formula(
        const std::string& formula,
        std::unordered_map<std::string, State>* var_map = nullptr);

template <typename Container>
bool evaluate_formula(
        const std::vector<std::string>& tokens,
        const Container& access_list);

template <typename Container>
bool evaluate_formula(
        const std::string& formula,
        const Container& access_list);

} // namespace complex_predicate
} // namespace faiss
