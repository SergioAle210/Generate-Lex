# Se importan las funciones de conversión a Postfix desde regexpToAFD.py
from regexpToAFD import toPostFix

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
# Sección 2: Funciones para expandir expresiones regulares
# ===============================


def is_boundary(ch):
    r"""
    Retorna True si el carácter no es alfanumérico ni guión bajo.
    """
    return not (ch.isalnum() or ch == "_")


def replace_whole_word(s, word, replacement):
    r"""
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
    r"""
    Expande el contenido de un conjunto entre corchetes, por ejemplo "0-9" o "a-zA-Z",
    a una alternancia: (0|1|...|9) o (a|b|...|z|A|B|...|Z).
    Si el contenido está entre comillas simples (por ejemplo, "'\n'"), se lo trata como literal.
    """
    content = content.strip()
    if content.startswith("'") and content.endswith("'"):
        literal = content[1:-1]
        return "(" + literal + ")"
    if content and content[0] == "\\":
        return "(" + content + ")"
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
    r"""
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
    r"""
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
    r"""
    Transforma en la expresión regular todos los operadores '+' (cerradura positiva)
    en la forma X+  -->  X (X)*, pero si el '+' ya está escapado (precedido de una barra invertida),
    se mantiene como literal.
    """

    def get_operand(expr: str, pos: int) -> (str, int):
        if pos <= 0:
            return "", 0
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
            operand = expr[j:pos]
            return operand, j
        else:
            return expr[pos - 1 : pos], pos - 1

    output = ""
    i = 0
    while i < len(expr):
        if expr[i] == "+":
            if i > 0 and expr[i - 1] == "\\":
                output += "+"
                i += 1
                continue
            operand, start_index = get_operand(expr, i)
            if operand == "":
                output += "+"
                i += 1
            else:
                output = output[: -len(operand)]
                transformed = operand + "(" + operand + ")*"
                output += transformed
                i += 1
        else:
            output += expr[i]
            i += 1
    return output


def escape_token_literals(expr: str) -> str:
    r"""
    Busca en la expresión subcadenas entre comillas simples.
    Si el contenido consiste en un solo carácter y es un operador especial (como +, *, (, )),
    lo reemplaza por su versión escapada (por ejemplo, '+' se transforma en "\+").
    """
    result = ""
    i = 0
    while i < len(expr):
        if expr[i] == "'":
            j = expr.find("'", i + 1)
            if j != -1:
                literal = expr[i + 1 : j]
                if len(literal) == 1 and literal in "+*()":
                    result += "\\" + literal
                else:
                    result += literal
                i = j + 1
            else:
                result += expr[i]
                i += 1
        else:
            result += expr[i]
            i += 1
    return result


def remove_outer_parentheses(expr: str) -> str:
    r"""
    Elimina recursivamente paréntesis exteriores redundantes.
    Si el contenido interno es exactamente un literal escapado (por ejemplo, "\(" o "\)"),
    se remueven los paréntesis y se retorna el literal.
    """
    if expr.startswith("(") and expr.endswith(")"):
        inner = expr[1:-1]
        # Si el contenido interno es exactamente un literal escapado, devuelve el literal sin los paréntesis
        if len(inner) == 2 and inner[0] == "\\" and inner[1] in "(" or inner[1] in ")":
            return inner
        count = 0
        for i, ch in enumerate(expr):
            if ch == "(":
                count += 1
            elif ch == ")":
                count -= 1
            # Si se cierra antes del final, los paréntesis exteriores no son redundantes
            if count == 0 and i < len(expr) - 1:
                return expr
        return remove_outer_parentheses(inner)
    return expr


def split_top_level(expr: str) -> list:
    r"""
    Divide la expresión en partes separadas por '|' a nivel superior.
    Tiene en cuenta secuencias escapadas (se omite el backslash y el siguiente carácter).
    """
    parts = []
    current = ""
    level = 0
    i = 0
    while i < len(expr):
        if expr[i] == "\\":
            # Añade la secuencia completa sin procesar
            if i + 1 < len(expr):
                current += expr[i : i + 2]
                i += 2
                continue
            else:
                current += expr[i]
                i += 1
                continue
        elif expr[i] == "(":
            level += 1
        elif expr[i] == ")":
            level -= 1
        if expr[i] == "|" and level == 0:
            parts.append(current)
            current = ""
        else:
            current += expr[i]
        i += 1
    if current:
        parts.append(current)
    return parts


def simplify_expression(expr: str) -> str:
    r"""
    Reduce paréntesis redundantes en cada parte de la expresión a nivel superior.
    """
    parts = split_top_level(expr)
    simplified_parts = [remove_outer_parentheses(part.strip()) for part in parts]
    return "|".join(simplified_parts)


# ===============================
# Sección 3: Ejemplo de uso con tokens reales del .yal
# ===============================

if __name__ == "__main__":
    # Procesa el archivo YALex real (ajusta la ruta de 'lexer.yal' según corresponda)
    result = parse_yalex("lexer.yal")

    # Construye la expresión regular unificada a partir de las alternativas de tokens
    token_alternatives = [rule for rule, act in result["rules"]]
    combined_expr = "(" + ")|(".join(token_alternatives) + ")"

    # Expande la expresión usando las definiciones extraídas
    expr_expandida = expand_regex(combined_expr, result["definitions"])
    # Se aplica escape a literales, se convierte el operador '+' y se simplifica la expresión
    expr_escapada = escape_token_literals(expr_expandida)
    expr_convertida = convert_plus_operator(expr_escapada)
    expr_simplificada = simplify_expression(expr_convertida)

    # Convierte la expresión final a Postfix
    postfix = toPostFix(expr_simplificada)

    # Se muestran sólo los resultados finales
    print("Expresión final simplificada:")
    print(expr_simplificada)
    print("Postfix generado:")
    print(postfix)
