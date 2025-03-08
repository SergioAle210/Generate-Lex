import pickle
from regexpToAFD import toPostFix, build_syntax_tree, construct_afd, minimize_afd


def parse_yalex(file_path):
    """Parsea un archivo YALex y extrae las definiciones y reglas."""
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    definitions = {}
    rules = []
    parsing_rules = False

    for line in lines:
        line = line.strip()
        if line.startswith("let "):  # Definición de una expresión regular
            parts = line.split("=")
            ident = parts[0].replace("let", "").strip()
            regex = parts[1].strip()
            definitions[ident] = regex
        elif line.startswith("rule "):  # Inicio de reglas
            parsing_rules = True
        elif parsing_rules and "|" in line:  # Reglas de tokens
            rule_parts = line.split("{")
            regex = rule_parts[0].strip()
            action = rule_parts[1].replace("}", "").strip()
            rules.append((regex, action))

    return definitions, rules


def expand_regex(regex, definitions):
    """Expande identificadores en una expresión regular usando las definiciones."""
    for ident, value in definitions.items():
        regex = regex.replace(ident, f"({value})")
    return regex


def build_global_regex(definitions, rules):
    """Construye una única expresión regular combinada con identificadores de tokens."""
    expanded_rules = [
        (expand_regex(regex, definitions).strip(), action)
        for regex, action in rules
        if regex.strip()
    ]

    # Elimina paréntesis innecesarios y ajusta correctamente las expresiones
    formatted_rules = []
    for i, (regex, _) in enumerate(expanded_rules):
        clean_regex = regex.strip()
        if not clean_regex.startswith("("):  # Evita doble paréntesis
            clean_regex = f"({clean_regex})"
        formatted_rules.append(f"{clean_regex}#{i+1}")

    global_regex = " | ".join(formatted_rules)  # Une correctamente con OR
    return global_regex, {i + 1: action for i, (_, action) in enumerate(expanded_rules)}


def save_afd(afd, filename="afd.pkl"):
    """Guarda el AFD en un archivo usando pickle."""
    with open(filename, "wb") as file:
        pickle.dump(afd, file)


def main():
    yalex_file = "lexer.yal"
    definitions, rules = parse_yalex(yalex_file)
    global_regex, token_map = build_global_regex(definitions, rules)
    postfix = toPostFix(global_regex)

    print(f"Expresión regular global: {global_regex}")
    print(f"Postfix generado: {postfix}")

    syntax_tree, position_symbol_map = build_syntax_tree(postfix)
    states, transitions, accepting_states = construct_afd(
        syntax_tree, position_symbol_map
    )
    mini_states, mini_transitions, mini_accepting_states, mini_initial_state = (
        minimize_afd(states, transitions, accepting_states)
    )

    afd_data = {
        "states": mini_states,
        "transitions": mini_transitions,
        "accepting_states": mini_accepting_states,
        "initial_state": mini_initial_state,
        "token_map": token_map,
    }
    save_afd(afd_data)
    print("AFD generado y guardado correctamente.")


if __name__ == "__main__":
    main()
