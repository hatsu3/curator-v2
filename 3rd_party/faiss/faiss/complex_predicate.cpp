#include <iostream>
#include <sstream>
#include <stack>

#include <faiss/complex_predicate.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace complex_predicate {

StateNode::StateNode(
        Type type,
        const Buffer& short_list,
        const Buffer& exclude_list)
        : type(type), short_list(short_list), exclude_list(exclude_list) {
    if (type != Type::SOME) {
        FAISS_ASSERT(short_list.size() == 0);
    }

    if (type != Type::MOST) {
        FAISS_ASSERT(exclude_list.size() == 0);
    }

    if (type == Type::MOST && exclude_list.empty()) {
        this->type = Type::ALL;
    }

    if (type == Type::SOME && short_list.empty()) {
        this->type = Type::NONE;
    }
}

StateNode::StateNode(Type type, Buffer&& short_list, Buffer&& exclude_list)
        : type(type),
          short_list(std::move(short_list)),
          exclude_list(std::move(exclude_list)) {
    if (type != Type::SOME) {
        FAISS_ASSERT(this->short_list.size() == 0);
    }

    if (type != Type::MOST) {
        FAISS_ASSERT(this->exclude_list.size() == 0);
    }

    if (type == Type::MOST && this->exclude_list.empty()) {
        this->type = Type::ALL;
    }

    if (type == Type::SOME && this->short_list.empty()) {
        this->type = Type::NONE;
    }
}

std::string StateNode::type_to_str() const {
    switch (type) {
        case Type::NONE:
            return "NONE";
        case Type::SOME:
            return "SOME";
        case Type::MOST:
            return "MOST";
        case Type::ALL:
            return "ALL";
        default:
            return "UNKNOWN";
    }
}

std::ostream& operator<<(std::ostream& os, const StateNode& state) {
    os << "State(Type: " << state.type_to_str() << ", Short List: [";

    for (size_t i = 0; i < state.short_list.size(); ++i) {
        os << state.short_list[i]
           << (i < state.short_list.size() - 1 ? ", " : "");
    }

    os << "], Exclude List: [";

    for (size_t i = 0; i < state.exclude_list.size(); ++i) {
        os << state.exclude_list[i]
           << (i < state.exclude_list.size() - 1 ? ", " : "");
    }

    os << "])";
    return os;
}

State make_state(Type type, const Buffer& list) {
    switch (type) {
        case Type::NONE:
            return std::make_shared<StateNode>(type, Buffer{}, Buffer{});
        case Type::SOME:
            return std::make_shared<StateNode>(type, list, Buffer{});
        case Type::MOST:
            return std::make_shared<StateNode>(type, Buffer{}, list);
        case Type::ALL:
            return std::make_shared<StateNode>(type, Buffer{}, Buffer{});
        default:
            return std::make_shared<StateNode>(
                    Type::UNKNOWN, Buffer{}, Buffer{});
    }
}

VarMapNode::VarMapNode(std::unordered_map<std::string, State>&& var_map)
        : var_map(std::move(var_map)) {}

VarMap VarMapNode::update(const std::string& name, State new_state) const {
    VarMap new_var_map = std::make_shared<VarMapNode>(*this);
    new_var_map->var_map[name] = new_state;
    return new_var_map;
}

VarMap VarMapNode::update(
        const std::unordered_map<std::string, State>& new_var_map) const {
    VarMap updated_var_map = std::make_shared<VarMapNode>(*this);
    for (auto& [name, state] : new_var_map) {
        updated_var_map->var_map[name] = state;
    }
    return updated_var_map;
}

const State& VarMapNode::get(const std::string& name) const {
    if (var_map.find(name) == var_map.end()) {
        throw std::runtime_error("Variable " + name + " not found");
    }
    return var_map.at(name);
}

std::vector<std::string> VarMapNode::unresolved_vars() const {
    std::vector<std::string> vars;
    for (auto& [name, state] : var_map) {
        if (*state == Type::UNKNOWN) {
            vars.push_back(name);
        }
    }
    return vars;
}

std::ostream& operator<<(std::ostream& os, const VarMapNode& var_map) {
    os << "VarMap(";
    for (auto it = var_map.var_map.begin(); it != var_map.var_map.end(); ++it) {
        os << it->first << ": " << *it->second;
        if (std::next(it) != var_map.var_map.end()) {
            os << ", ";
        }
    }
    os << ")";
    return os;
}

State VariableNode::evaluate(VarMap var_map) const {
    return var_map->get(var_name);
}

State NotNode::evaluate(VarMap var_map) const {
    State op = operand->evaluate(var_map);

    switch (op->type) {
        case Type::NONE:
            return make_state(Type::ALL);
        case Type::SOME:
            return make_state(Type::MOST, op->short_list);
        case Type::MOST:
            return make_state(Type::SOME, op->exclude_list);
        case Type::ALL:
            return make_state(Type::NONE);
        default:
            return make_state(Type::UNKNOWN);
    }
}

State AndNode::evaluate(VarMap var_map) const {
    State lhs = left->evaluate(var_map);
    State rhs = right->evaluate(var_map);

    // If either side is NONE, the value of the expression is NONE
    if (*lhs == Type::NONE || *rhs == Type::NONE) {
        return make_state(Type::NONE);
    }

    // If either side is ALL, the value of the expression is the other side
    if (*lhs == Type::ALL) {
        return rhs;
    } else if (*rhs == Type::ALL) {
        return lhs;
    }

    // If we have an UNKNOWN state, we can't infer anything
    if (*lhs == Type::UNKNOWN || *rhs == Type::UNKNOWN) {
        return make_state(Type::UNKNOWN);
    }

    // Otherwise, we have two SOME or MOST states
    if (*lhs == Type::SOME && *rhs == Type::SOME) {
        auto short_list = buffer_intersect(lhs->short_list, rhs->short_list);
        return make_state(Type::SOME, std::move(short_list));
    } else if (*lhs == Type::MOST && *rhs == Type::MOST) {
        auto exclude_list = buffer_union(lhs->exclude_list, rhs->exclude_list);
        return make_state(Type::MOST, std::move(exclude_list));
    } else if (*lhs == Type::SOME && *rhs == Type::MOST) {
        auto short_list = buffer_difference(lhs->short_list, rhs->exclude_list);
        return make_state(Type::SOME, std::move(short_list));
    } else {
        auto short_list = buffer_difference(rhs->short_list, lhs->exclude_list);
        return make_state(Type::SOME, std::move(short_list));
    }
}

State OrNode::evaluate(VarMap var_map) const {
    State lhs = left->evaluate(var_map);
    State rhs = right->evaluate(var_map);

    // If either side is ALL, the value of the expression is ALL
    if (*lhs == Type::ALL || *rhs == Type::ALL) {
        return make_state(Type::ALL);
    }

    // If either side is NONE, the value of the expression is the other side
    if (*lhs == Type::NONE) {
        return rhs;
    } else if (*rhs == Type::NONE) {
        return lhs;
    }

    // If we have an UNKNOWN state, we can't infer anything
    if (*lhs == Type::UNKNOWN || *rhs == Type::UNKNOWN) {
        return make_state(Type::UNKNOWN);
    }

    // Otherwise, we have two SOME or MOST states
    if (*lhs == Type::SOME && *rhs == Type::SOME) {
        auto short_list = buffer_union(lhs->short_list, rhs->short_list);
        return make_state(Type::SOME, std::move(short_list));
    } else if (*lhs == Type::MOST && *rhs == Type::MOST) {
        auto exclude_list =
                buffer_intersect(lhs->exclude_list, rhs->exclude_list);
        return make_state(Type::MOST, std::move(exclude_list));
    } else if (*lhs == Type::SOME && *rhs == Type::MOST) {
        auto exclude_list =
                buffer_difference(rhs->exclude_list, lhs->short_list);
        return make_state(Type::MOST, std::move(exclude_list));
    } else {
        auto exclude_list =
                buffer_difference(lhs->exclude_list, rhs->short_list);
        return make_state(Type::MOST, std::move(exclude_list));
    }
}

std::vector<std::string> tokenize_formula(const std::string& formula) {
    std::istringstream iss(formula);
    std::vector<std::string> tokens;
    std::string token;

    while (iss >> token) {
        tokens.push_back(token);
    }

    return tokens;
}

Expr parse_formula(
        const std::string& formula,
        std::unordered_map<std::string, State>* var_map) {
    auto tokens = tokenize_formula(formula);
    std::stack<Expr> stack;

    for (auto it = tokens.rbegin(); it != tokens.rend(); ++it) {
        auto& token = *it;
        if (token == "AND" || token == "OR") {
            auto right = std::move(stack.top());
            stack.pop();
            auto left = std::move(stack.top());
            stack.pop();
            if (token == "AND") {
                stack.push(make_and(std::move(left), std::move(right)));
            } else {
                stack.push(make_or(std::move(left), std::move(right)));
            }
        } else if (token == "NOT") {
            auto operand = std::move(stack.top());
            stack.pop();
            stack.push(make_not(std::move(operand)));
        } else {
            stack.push(make_var(token));
            if (var_map) {
                var_map->emplace(token, make_state(Type::UNKNOWN));
            }
        }
    }

    return std::move(stack.top());
}

template <typename Container>
bool evaluate_formula(
        const std::vector<std::string>& tokens,
        const Container& access_list) {
    std::vector<bool> stack;

    auto eval_token = [&access_list](const std::string& token) -> bool {
        try {
            int num = std::stoi(token);
            return std::find(access_list.begin(), access_list.end(), num) !=
                    access_list.end();
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument("Malformed formula: " + token);
        }
    };

    for (auto it = tokens.rbegin(); it != tokens.rend(); ++it) {
        const std::string& tok = *it;
        if (tok == "AND" || tok == "OR" || tok == "NOT") {
            if (tok == "AND") {
                if (stack.size() < 2)
                    throw std::runtime_error("Insufficient operands for AND");
                bool a = stack.back();
                stack.pop_back();
                bool b = stack.back();
                stack.pop_back();
                stack.push_back(a && b);
            } else if (tok == "OR") {
                if (stack.size() < 2)
                    throw std::runtime_error("Insufficient operands for OR");
                bool a = stack.back();
                stack.pop_back();
                bool b = stack.back();
                stack.pop_back();
                stack.push_back(a || b);
            } else if (tok == "NOT") {
                if (stack.empty())
                    throw std::runtime_error("Insufficient operands for NOT");
                bool a = stack.back();
                stack.pop_back();
                stack.push_back(!a);
            }
        } else {
            stack.push_back(eval_token(tok));
        }
    }

    if (stack.size() != 1) {
        throw std::runtime_error("Malformed expression");
    }

    return stack.back();
}

template <typename Container>
bool evaluate_formula(
        const std::string& formula,
        const Container& access_list) {
    auto tokens = tokenize_formula(formula);
    return evaluate_formula(tokens, access_list);
}

template bool evaluate_formula(
        const std::vector<std::string>& tokens,
        const std::unordered_set<tid_t>& access_list);

template bool evaluate_formula(
        const std::vector<std::string>& tokens,
        const std::vector<tid_t>& access_list);

template bool evaluate_formula(
        const std::string& formula,
        const std::unordered_set<tid_t>& access_list);

template bool evaluate_formula(
        const std::string& formula,
        const std::vector<tid_t>& access_list);

} // namespace complex_predicate
} // namespace faiss