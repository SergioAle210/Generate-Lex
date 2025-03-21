# Se importan las funciones de conversión a Postfix desde regexpToAFD.py
from regexpToAFD import toPostFix, build_syntax_tree, construct_afd, print_afd
from regex_transform import (
    expand_regex, 
    escape_token_literals, 
    expand_bracket_ranges, 
    convert_plus_operator, 
    convert_optional_operator, 
    simplify_expression, 
    attach_markers_to_final_regexp,
    process_difference_operator,
    balance_parentheses
)
# ===============================
# Sección 1: Funciones para parsear YALex
# ===============================


def remove_comments_yalex(text):
    r"""
    Elimina todas las ocurrencias de comentarios delimitados por '(*' y '*)'.
    Se asume que los comentarios no están anidados.
    """
    while True:
        start = text.find("(*")
        if start == -1:
            break
        end = text.find("*)", start + 2)
        if end == -1:
            text = text[:start]
            break
        text = text[:start] + text[end + 2 :]
    return text


def extract_header_and_trailer(text):
    r"""
    Extrae el bloque {header} al inicio y el bloque {trailer} al final (si existen).
    """
    header = ""
    trailer = ""
    text = text.strip()
    if text.startswith("{"):
        end = text.find("}")
        if end != -1:
            header = text[1:end].strip()
            text = text[end + 1 :].strip()
    last_open = text.rfind("{")
    last_close = text.rfind("}")
    if last_open != -1 and last_close != -1 and last_close > last_open:
        trailer = text[last_open + 1 : last_close].strip()
        text = text[:last_open].strip()
    return header, trailer, text


def extract_definitions(text):
    r"""
    Extrae las definiciones "let ident = regexp" línea por línea.
    Retorna un diccionario con las definiciones y el texto sin esas líneas.
    """
    definitions = {}
    lines = text.splitlines()
    new_lines = []
    for line in lines:
        line_strip = line.strip()
        if line_strip.startswith("let "):
            sin_let = line_strip[4:]
            eq_index = sin_let.find("=")
            if eq_index != -1:
                ident = sin_let[:eq_index].strip()
                regexp = sin_let[eq_index + 1 :].strip()
                definitions[ident] = regexp
        else:
            new_lines.append(line)
    text_without_defs = "\n".join(new_lines)
    return definitions, text_without_defs


def extract_rule(text):
    r"""
    Extrae la sección 'rule entrypoint [...] =' y retorna el nombre del entrypoint y el cuerpo de la regla.
    """
    idx = text.find("rule ")
    if idx == -1:
        return "", ""
    end_line = text.find("\n", idx)
    if end_line == -1:
        end_line = len(text)
    rule_header = text[idx:end_line].strip()
    parts = rule_header.split()
    entrypoint_name = parts[1] if len(parts) > 1 else ""
    rule_body = text[end_line:].strip()
    return entrypoint_name, rule_body


def extract_token_rules(rule_body):
    r"""
    Separa las alternativas del cuerpo de la regla, asumiendo que están separadas por '|'.
    Para cada alternativa, extrae el bloque de acción (entre '{' y '}') si existe.
    Retorna una lista de tuplas (regexp, action).
    """
    token_rules = []
    alternatives = rule_body.split("|")
    for alt in alternatives:
        alt = alt.strip()
        if not alt:
            continue
        open_brace = alt.find("{")
        if open_brace != -1:
            regexp_part = alt[:open_brace].strip()
            close_brace = alt.find("}", open_brace)
            action_part = (
                alt[open_brace + 1 : close_brace].strip() if close_brace != -1 else ""
            )
            token_rules.append((regexp_part, action_part))
        else:
            token_rules.append((alt, ""))
    return token_rules


def parse_yalex(filepath):
    r"""
    Procesa un archivo YALex y retorna un diccionario con:
      - header, trailer, definitions, entrypoint y rules.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    content = remove_comments_yalex(content)
    header, trailer, remaining = extract_header_and_trailer(content)
    definitions, remaining = extract_definitions(remaining)
    entrypoint, rule_body = extract_rule(remaining)
    token_rules = extract_token_rules(rule_body)
    return {
        "header": header,
        "trailer": trailer,
        "definitions": definitions,
        "entrypoint": entrypoint,
        "rules": token_rules,
    }



# ===============================
# Sección 3: Ejemplo de uso con tokens reales del .yal
# ===============================
# En el archivo principal, asegúrate de que los tokens con diferencia se procesen correctamente
if __name__ == "__main__":
    # Procesa el archivo YALex real
    result = parse_yalex("lexer.yal")

    # Process each token alternative separately for better debugging
    processed_alternatives = []
    token_actions = []
    
    print("Processing individual tokens:")
    for i, (rule, action) in enumerate(result["rules"]):
        print(f"\nToken {i+1}: {rule}")
        
        try:
            # Process step by step but with more concise output
            expanded = expand_regex(rule, result["definitions"])
            print(f"  Expansion: {expanded[:50]}{'...' if len(expanded) > 50 else ''}")
            
            escaped = escape_token_literals(expanded)
            # Only show differences in output if they exist
            if escaped != expanded:
                print(f"  Escaped: {escaped[:50]}{'...' if len(escaped) > 50 else ''}")
            
            # Process difference operator
            diff_processed = process_difference_operator(escaped)
            if diff_processed != escaped:
                print(f"  Diff processed: {diff_processed[:50]}{'...' if len(diff_processed) > 50 else ''}")
            elif "#" in escaped:
                print("  WARNING: Difference operator not processed correctly")
            
            ranges_expanded = expand_bracket_ranges(diff_processed)
            if ranges_expanded != diff_processed:
                print(f"  Ranges expanded: {ranges_expanded[:50]}{'...' if len(ranges_expanded) > 50 else ''}")
            
            plus_converted = convert_plus_operator(ranges_expanded)
            if plus_converted != ranges_expanded:
                print(f"  Plus converted: {plus_converted[:50]}{'...' if len(plus_converted) > 50 else ''}")
            
            optional_converted = convert_optional_operator(plus_converted)
            if optional_converted != plus_converted:
                print(f"  Optional converted: {optional_converted[:50]}{'...' if len(optional_converted) > 50 else ''}")
            
            # Balance parentheses
            balanced = balance_parentheses(optional_converted)
            if balanced != optional_converted:
                print(f"  Balanced: {balanced[:50]}{'...' if len(balanced) > 50 else ''}")
            
            simplified = simplify_expression(balanced)
            
            # Special cases handling
            if rule == "'('":
                simplified = "\\("
            elif rule == "')'":
                simplified = "\\)"
            elif rule == "'+'":
                simplified = "\\+"
            elif rule == "'*'":
                simplified = "\\*"
            
            # Final result always shown
            print(f"  Final: {simplified[:50]}{'...' if len(simplified) > 50 else ''}")
            
            # Group if needed
            if "|" in simplified and not (simplified.startswith("(") and simplified.endswith(")")):
                simplified = "(" + simplified + ")"
                print(f"  Grouped: {simplified[:50]}{'...' if len(simplified) > 50 else ''}")
            
            # Final validation
            simplified = balance_parentheses(simplified)
            
            # Only add non-empty expressions
            if simplified.strip():
                processed_alternatives.append(simplified)
                token_actions.append(action)
            else:
                print("  WARNING: Empty expression detected and skipped")
        except Exception as e:
            print(f"  ERROR processing token: {str(e)}")
            # Add a placeholder to maintain token order
            processed_alternatives.append("ERROR")
            token_actions.append(action)
    
    # After processing all tokens and combining them
    # Create a mapping from markers to actions
    marker_action_map = {}
    
    # Combine only valid alternatives
    valid_alternatives = [alt for alt in processed_alternatives if alt != "ERROR"]
    valid_actions = [action for i, action in enumerate(token_actions) if processed_alternatives[i] != "ERROR"]
    
    # Join alternatives with |
    combined_expr = "|".join(valid_alternatives)
    
    # Validar paréntesis balanceados en la expresión combinada
    combined_expr = balance_parentheses(combined_expr)
    
    # Adjunta los identificadores únicos a cada alternativa
    final_expr, marker_mapping = attach_markers_to_final_regexp(
        combined_expr, start_id=1000
    )
    
    # Create a mapping from markers to actions
    for i, (marker, pattern) in enumerate(marker_mapping.items()):
        if i < len(valid_actions):
            marker_action_map[marker] = valid_actions[i]
        else:
            marker_action_map[marker] = "/* No action specified */"
    
    # Reemplaza los separadores especiales
    final_expr = final_expr.replace("§", "|")
    
    # Convierte la expresión final (con marcadores) a Postfix
    postfix = toPostFix(final_expr)
    
    # Al guardar en el archivo, asegúrate de incluir toda la información
    with open("c:\\Users\\rodri\\Documents\\Diseño-Lenguajes\\Generate-Lex\\marker_actions.txt", "w", encoding="utf-8") as f:
        f.write("=== MARKER TO ACTION MAPPING ===\n\n")
        for marker, action in marker_action_map.items():
            f.write(f"Marker {marker}: {action}\n")
        
        f.write("\n=== MARKER TO PATTERN MAPPING ===\n\n")
        for marker, pattern in marker_mapping.items():
            f.write(f"Marker {marker}: {pattern}\n")

    # En la sección de resultados finales, reemplaza el código que muestra solo los primeros 5 marcadores
    
    # Reemplazar esto:
    print("\nMapping de marcadores (primeros 5):")
    marker_items = list(marker_mapping.items())[:5]
    print(marker_items)
    if len(marker_mapping) > 5:
        print(f"... y {len(marker_mapping) - 5} más")
    
    print("\nMapping de marcadores a acciones (primeros 5):")
    action_items = list(marker_action_map.items())[:5]
    print(action_items)
    if len(marker_action_map) > 5:
        print(f"... y {len(marker_action_map) - 5} más")
    
    # Con esto:
    print("\nMapping completo de marcadores:")
    for marker, pattern in marker_mapping.items():
        print(f"Marker {marker}: {pattern[:50]}{'...' if len(pattern) > 50 else ''}")
    
    print("\nMapping completo de marcadores a acciones:")
    for marker, action in marker_action_map.items():
        print(f"Marker {marker}: {action}")
    
    # Simplified diagnostic information
    print("\nDiagnostic summary:")
    print(f"- Expression length: {len(final_expr)} chars")
    print(f"- Postfix length: {len(postfix)} chars")
    print(f"- Number of tokens: {len(marker_mapping)}")
    
    # Detailed token information
    print("\nDetailed token information:")
    for token_id, (token_pattern, token_action) in enumerate(zip(processed_alternatives, token_actions)):
        if token_pattern != "ERROR":
            print(f"Token {token_id+1}:")
            print(f"  Pattern: {token_pattern[:50]}{'...' if len(token_pattern) > 50 else ''}")
            print(f"  Action: {token_action[:50] if token_action else 'None'}")
            # Find the marker ID for this token
            marker_id = None
            for marker, pattern in marker_mapping.items():
                if pattern == token_pattern:
                    marker_id = marker
                    break
            print(f"  Marker ID: {marker_id}")
    
    # Skip the automaton and tree construction
    print("\nSkipping automaton and tree construction as requested.")
    
    # Optional: Save the processed data to files for later use
    try:
        with open("c:\\Users\\rodri\\Documents\\Diseño-Lenguajes\\Generate-Lex\\processed_regex.txt", "w", encoding="utf-8") as f:
            f.write("=== PROCESSED REGULAR EXPRESSIONS ===\n\n")
            for i, (pattern, action) in enumerate(zip(processed_alternatives, token_actions)):
                f.write(f"Token {i+1}:\n")
                f.write(f"Pattern: {pattern}\n")
                f.write(f"Action: {action}\n\n")
            
            f.write("=== COMBINED EXPRESSION ===\n\n")
            f.write(f"{combined_expr}\n\n")
            
            f.write("=== FINAL EXPRESSION WITH MARKERS ===\n\n")
            f.write(f"{final_expr}\n\n")
            
            f.write("=== POSTFIX EXPRESSION ===\n\n")
            f.write(f"{postfix}\n\n")
            
            f.write("=== MARKER MAPPING ===\n\n")
            for marker, pattern in marker_mapping.items():
                f.write(f"Marker {marker}: {pattern}\n")
        
        print("\nProcessed data saved to 'processed_regex.txt'")
    except Exception as e:
        print(f"\nError saving processed data: {e}")

    # Try to build the syntax tree with better error handling
    try:
        print("\nBuilding syntax tree...")
        root, position_symbol_map = build_syntax_tree(postfix)
        print("✓ Syntax tree construction successful")
        
        print("\nConstructing AFD...")
        afd = construct_afd(root, position_symbol_map)
        print("✓ AFD construction successful")
        
        # Only print a summary of the AFD, not the full details
        print(f"\nAFD Summary:")
        # Check if afd is a tuple or a dictionary
        if isinstance(afd, tuple):
            # If it's a tuple, unpack it properly
            states, transitions, accepting_states = afd
            print(f"- States: {len(states)}")
            print(f"- Transitions: {len(transitions)}")
            print(f"- Accepting states: {len(accepting_states)}")
        elif isinstance(afd, dict):
            # If it's a dictionary, access it with keys
            print(f"- States: {len(afd['states'])}")
            print(f"- Transitions: {len(afd['transitions'])}")
            print(f"- Accepting states: {len(afd['accepting_states'])}")
        else:
            print(f"- AFD type: {type(afd)}")
            print(f"- AFD content: {afd}")
        
    except Exception as e:
        import traceback
        print(f"\n✗ Error: {e}")
        
        # Simplified error analysis
        print("\nAnalyzing expression:")
        operators = ['|', '.', '*', '+', '?']
        operand_count = sum(1 for c in postfix if c not in operators)
        operator_count = sum(1 for c in postfix if c in operators)
        
        print(f"- Operands: {operand_count}")
        print(f"- Operators: {operator_count}")
        
        if operator_count >= operand_count:
            print("- Issue: More operators than operands")
        
        # Show only the first few lines of the stack trace
        tb_lines = traceback.format_exc().split('\n')
        print("\nStack trace (first 3 lines):")
        for line in tb_lines[:3]:
            print(line)
    