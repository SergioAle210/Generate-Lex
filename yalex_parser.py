# ===============================
# Sección 1: Funciones para parsear YALex
# ===============================


def remove_comments_yalex(text):
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
# Sección 2: Funciones para expandir expresiones regulares
# ===============================


def is_boundary(ch):
    """
    Retorna True si el carácter no es alfanumérico ni guión bajo.
    """
    return not (ch.isalnum() or ch == "_")


def replace_whole_word(s, word, replacement):
    """
    Reemplaza en 's' cada ocurrencia de 'word' delimitada por límites por 'replacement'.
    Retorna el nuevo string y el número de reemplazos realizados.
    """
    result = ""
    i = 0
    count = 0
    while i < len(s):
        if s[i : i + len(word)] == word:
            prev_char = s[i - 1] if i > 0 else None
            next_char = s[i + len(word)] if i + len(word) < len(s) else None
            if (prev_char is None or is_boundary(prev_char)) and (
                next_char is None or is_boundary(next_char)
            ):
                result += replacement
                i += len(word)
                count += 1
                continue
        result += s[i]
        i += 1
    return result, count


def expand_bracket_content(content):
    """
    Expande el contenido de un conjunto entre corchetes, por ejemplo "0-9" o "a-zA-Z",
    a una alternancia: (0|1|...|9) o (a|b|...|z|A|B|...|Z).
    """
    expanded_chars = []
    i = 0
    while i < len(content):
        if i + 2 < len(content) and content[i + 1] == "-":
            start_char = content[i]
            end_char = content[i + 2]
            for code in range(ord(start_char), ord(end_char) + 1):
                expanded_chars.append(chr(code))
            i += 3
        else:
            expanded_chars.append(content[i])
            i += 1
    return "(" + "|".join(expanded_chars) + ")"


def expand_bracket_ranges(s):
    """
    Reemplaza en s las expresiones entre corchetes '[' y ']' por su expansión.
    Por ejemplo, "[0-9]" se convierte en "(0|1|...|9)".
    """
    result = ""
    i = 0
    while i < len(s):
        if s[i] == "[":
            j = s.find("]", i + 1)
            if j == -1:
                result += s[i]
                i += 1
            else:
                content = s[i + 1 : j]
                expanded = expand_bracket_content(content)
                result += expanded
                i = j + 1
        else:
            result += s[i]
            i += 1
    return result


def expand_regex(regexp, definitions):
    """
    Expande recursivamente la expresión regular:
      1. Reemplaza identificadores (definiciones) por su definición, delimitándolos con paréntesis.
      2. Expande los conjuntos tipo [0-9] o [a-zA-Z] a una alternancia explícita.
    """
    changed = True
    result = regexp
    while changed:
        changed = False
        for ident, definicion in definitions.items():
            replacement = "(" + definicion + ")"
            new_result, count = replace_whole_word(result, ident, replacement)
            if count > 0:
                changed = True
                result = new_result
    result = expand_bracket_ranges(result)
    return result


def convert_plus_operator(expr: str) -> str:
    """
    Transforma en la expresión regular todos los operadores '+' (cerradura positiva)
    en la forma X+  -->  X (X)*.

    Para ello, se busca el operando inmediatamente anterior al '+'.
      - Si el operando es un grupo (delimitado por paréntesis), se toma todo el grupo.
      - Si es un solo carácter, se toma ese carácter.
    """

    def get_operand(expr: str, pos: int) -> (str, int):
        """
        Retorna una tupla (operand, start_index) donde 'operand' es la subcadena
        que actúa como operando del '+' que se encuentra en expr[pos], y start_index
        es la posición en expr donde inicia dicho operando.
        """
        if pos <= 0:
            return "", 0
        # Si el carácter inmediatamente anterior es ')', se busca el '(' que lo abre.
        if expr[pos - 1] == ")":
            count = 1
            j = pos - 2
            while j >= 0:
                if expr[j] == ")":
                    count += 1
                elif expr[j] == "(":
                    count -= 1
                    if count == 0:
                        break
                j -= 1
            operand = expr[j:pos]  # incluye desde '(' hasta ')'
            return operand, j
        else:
            # Operando de un solo carácter.
            return expr[pos - 1 : pos], pos - 1

    output = ""
    i = 0
    while i < len(expr):
        if expr[i] == "+":
            # Obtener el operando inmediatamente anterior.
            operand, start_index = get_operand(expr, i)
            # Remover el operando que ya fue agregado a la salida.
            output = output[: -len(operand)]
            # Transformar: X+  -->  X (X)*
            transformed = operand + "(" + operand + ")*"
            output += transformed
            i += 1  # Saltamos el '+'.
        else:
            output += expr[i]
            i += 1
    return output


# ===============================
# Sección 3: Conversión a Postfix
# ===============================
# Importamos las funciones de conversión a Postfix desde regexpToAFD.py
from regexpToAFD import toPostFix

# ===============================
# Sección 4: Ejemplo de uso con tokens reales del .yal
# ===============================

if __name__ == "__main__":
    print("=== Ejemplo de parseo YALex ===")
    filepath = (
        "lexer.yal"  # Asegúrate de tener tu archivo actualizado con los tokens reales
    )
    result = parse_yalex(filepath)

    print("Header:")
    print(result["header"])
    print("\nDefiniciones:")
    for k, v in result["definitions"].items():
        print(f"{k} = {v}")
    print("\nEntrypoint:")
    print(result["entrypoint"])
    print("\nReglas de tokens (regexp, action):")
    for regexp_rule, action in result["rules"]:
        print(f"Expresión: {regexp_rule}  |  Acción: {action}")
    print("\nTrailer:")
    print(result["trailer"])

    token_alternatives = [rule for rule, act in result["rules"]]

    combined_expr = "(" + ")|(".join(token_alternatives) + ")"
    print("\nExpresión unificada (antes de expansión):")
    print(combined_expr)

    expr_expandida = expand_regex(combined_expr, result["definitions"])
    print("\nExpresión unificada expandida:")
    print(expr_expandida)

    # Convertir los operadores '+' a la forma X (X)*.
    expr_convertida = convert_plus_operator(expr_expandida)
    print("\nExpresión tras conversión de '+':")
    print(expr_convertida)

    # Convertir la expresión resultante a Postfix.
    postfix = toPostFix(expr_convertida)
    print("\nPostfix generado:")
    print(postfix)
