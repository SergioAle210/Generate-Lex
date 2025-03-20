# Se importan las funciones de conversión a Postfix desde regexpToAFD.py
from regexpToAFD import toPostFix, build_syntax_tree, construct_afd, print_afd

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


def custom_escape_char(ch: str) -> str:
    """Escapa el carácter si es especial en regex, incluyendo la comilla simple."""
    if ch in ".^$*+?{}[]\\|()'":
        return "\\" + ch
    return ch


def custom_escape_str(s: str) -> str:
    """Escapa todos los caracteres de la cadena."""
    return "".join(custom_escape_char(c) for c in s)


def expand_bracket_content(content: str) -> str:
    r"""
    Expande el contenido de un conjunto entre corchetes, por ejemplo "0-9" o "a-zA-Z",
    a una alternancia: (0|1|...|9) o (a|b|...|z|A|B|...|Z).
    Si el contenido comienza con "^", se interpreta como un conjunto negado.
    Si el contenido está entre comillas simples (por ejemplo, "'\n'"), se lo trata como literal.
    """
    content = content.strip()
    if content.startswith("^"):
        # Caso de conjunto negado
        negated_content = content[1:]
        # Alfabeto de caracteres ASCII imprimibles (32 a 126), excluyendo '|'
        alphabet = "".join(chr(i) for i in range(32, 127) if chr(i) != "|")
        return expand_negated_bracket_content(negated_content, alphabet)
    # Se modifica la condición para que solo se considere literal si hay algo entre las comillas
    if content.startswith("'") and content.endswith("'"):
        literal = content[1:-1]
        return "(" + custom_escape_str(literal) + ")"
    if content and content[0] == "\\":
        return "(" + content + ")"
    expanded_chars = []
    i = 0
    while i < len(content):
        # Si se detecta un rango (por ejemplo, a-z)
        if i + 2 < len(content) and content[i + 1] == "-":
            start_char = content[i]
            end_char = content[i + 2]
            for code in range(ord(start_char), ord(end_char) + 1):
                expanded_chars.append(custom_escape_char(chr(code)))
            i += 3
        else:
            expanded_chars.append(custom_escape_char(content[i]))
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


def convert_optional_operator(expr: str) -> str:
    r"""
    Convierte el operador '?' (opcional) a su forma equivalente:
      R?  -->  (R§_)
    donde "§" es un separador especial (que luego se reemplazará por "|" en el resultado final)
    y "_" representa la cadena vacía.

    Se asume que '?' es un operador postfix aplicado al operando inmediatamente anterior.
    Si el '?' está escapado (precedido de "\"), se deja como literal.
    """
    output = ""
    i = 0
    while i < len(expr):
        if expr[i] == "?":
            if i > 0 and expr[i - 1] == "\\":
                output += "?"
                i += 1
                continue
            # Si el operando es un grupo (termina en ")")
            if output and output[-1] == ")":
                count = 1
                j = len(output) - 2
                while j >= 0:
                    if output[j] == ")":
                        count += 1
                    elif output[j] == "(":
                        count -= 1
                        if count == 0:
                            break
                    j -= 1
                operand = output[j:]  # desde el '(' correspondiente hasta el final
                output = output[:j]  # eliminamos el operando de output
                # Se genera el grupo opcional usando el separador especial "§"
                transformed = "(" + operand + "§_)"
                output += transformed
            else:
                # Caso: operando de un solo carácter
                if output:
                    operand = output[-1]
                    output = output[:-1]
                    transformed = "(" + operand + "§_)"
                    output += transformed
                else:
                    output += "§_"
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
                if len(literal) == 1 and literal in "+*()-/%":
                    result += "\\" + literal
                else:
                    result += literal
                i = j + 1
            else:
                result += expr[i]
                i += 1
        elif expr[i] == '"':
            j = expr.find('"', i + 1)
            if j != -1:
                literal = expr[i + 1 : j]
                if literal:
                    # Convierte el literal en una concatenación de caracteres
                    transformed = literal[0]
                    for ch in literal[1:]:
                        transformed += "." + ch
                    result += transformed
                else:
                    # Literal vacío, lo interpretamos como epsilon (se puede representar con "_" u otro símbolo según convenga)
                    result += "_"
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
        if len(inner) == 2 and inner[0] == "\\" and (inner[1] in ("(", ")")):
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


def attach_markers_to_final_regexp(expr: str, start_id=1000) -> (str, dict):
    """
    Dado un string 'expr' que es la unión de alternativas separadas por '|' a nivel superior,
    le adjunta un marcador único (un número a partir de start_id) al final de cada alternativa.
    Si una alternativa contiene un '|' (por ejemplo, por la conversión opcional que usó "§"),
    se envuelve en paréntesis para que se considere un único token.

    Retorna:
      - new_expr: La nueva expresión unificada con cada alternativa terminada en su marcador.
      - marker_mapping: Un diccionario que mapea cada marcador al literal correspondiente,
                        donde el separador especial "§" se reemplaza por "|".
    """
    parts = split_top_level(expr)
    new_parts = []
    marker_mapping = {}
    current_id = start_id
    for part in parts:
        stripped = part.strip()
        # Si la alternativa contiene un '|' y no está agrupada, la envolvemos en paréntesis.
        if "|" in stripped and not (
            stripped.startswith("(") and stripped.endswith(")")
        ):
            stripped = "(" + stripped + ")"
        new_part = stripped + str(current_id)
        new_parts.append(new_part)
        # En el mapping reemplazamos el separador especial "§" por "|"
        mapped_literal = stripped.replace("§", "|")
        marker_mapping[current_id] = mapped_literal
        current_id += 1
    new_expr = "|".join(new_parts)
    return new_expr, marker_mapping


def expand_negated_bracket_content(content: str, alphabet: str) -> str:
    """
    Expande el contenido de un conjunto negado, es decir, [^...].
    Calcula el complemento del conjunto definido en 'content' respecto a 'alphabet'
    y lo retorna como una alternancia usando el separador especial "¦":
      (a¦b¦c¦...).
    """
    expanded = set()
    i = 0
    while i < len(content):
        if i + 2 < len(content) and content[i + 1] == "-":
            start_char = content[i]
            end_char = content[i + 2]
            for code in range(ord(start_char), ord(end_char) + 1):
                expanded.add(chr(code))
            i += 3
        else:
            expanded.add(content[i])
            i += 1
    comp = sorted(set(alphabet) - expanded)
    # Escapamos cada carácter especial manualmente
    return "(" + "¦".join(custom_escape_char(c) for c in comp) + ")"


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
    expr_escapada = escape_token_literals(expr_expandida)
    expr_convertida = convert_plus_operator(expr_escapada)
    expr_optional = convert_optional_operator(expr_convertida)
    expr_simplificada = simplify_expression(expr_optional)

    # Adjunta los identificadores únicos a cada alternativa
    final_expr, marker_mapping = attach_markers_to_final_regexp(
        expr_simplificada, start_id=1000
    )

    # Reemplaza el separador especial "§" por "|" en la expresión final
    final_expr = final_expr.replace("§", "|")
    final_expr = final_expr.replace("¦", "|")

    # Convierte la expresión final (con marcadores) a Postfix
    postfix = toPostFix(final_expr)

    # Se muestran sólo los resultados finales
    print("Expresión final:")
    print(final_expr)
    print("Postfix generado:")
    print(postfix)
    print("Mapping de marcadores:")
    print(marker_mapping)
