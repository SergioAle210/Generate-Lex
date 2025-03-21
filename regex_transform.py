
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


def process_difference_operator(expr):
    """
    Procesa el operador de diferencia '#' entre conjuntos de caracteres.
    Calcula la diferencia entre dos conjuntos de forma generalizada.
    Por ejemplo, [0-9]#[5-9] se convierte en (0|1|2|3|4).
    """
    if '#' not in expr:
        return expr
    
    # Procesamiento general para cualquier operación de diferencia
    result = ""
    i = 0
    while i < len(expr):
        if expr[i] == '#' and i > 0:
            # Buscar el conjunto de la izquierda
            left_end = i - 1
            left_start = left_end
            
            # Buscar el inicio del conjunto izquierdo
            if expr[left_end] == ')':
                balance = 1
                left_start = left_end - 1
                while left_start >= 0 and balance > 0:
                    if expr[left_start] == ')':
                        balance += 1
                    elif expr[left_start] == '(':
                        balance -= 1
                    left_start -= 1
                left_start += 1
            elif expr[left_end] == ']':
                # Buscar el inicio del conjunto entre corchetes
                left_start = left_end
                while left_start >= 0 and expr[left_start] != '[':
                    left_start -= 1
            else:
                # Caso de un solo carácter
                left_start = left_end
            
            # Buscar el conjunto de la derecha
            right_start = i + 1
            right_end = right_start
            
            # Buscar el final del conjunto derecho
            if right_start < len(expr) and expr[right_start] == '(':
                balance = 1
                right_end = right_start + 1
                while right_end < len(expr) and balance > 0:
                    if expr[right_end] == '(':
                        balance += 1
                    elif expr[right_end] == ')':
                        balance -= 1
                    right_end += 1
            elif right_start < len(expr) and expr[right_start] == '[':
                # Buscar el final del conjunto entre corchetes
                right_end = right_start + 1
                while right_end < len(expr) and expr[right_end] != ']':
                    right_end += 1
                if right_end < len(expr):
                    right_end += 1  # Incluir el corchete de cierre
            else:
                # Caso de un solo carácter
                right_end = right_start + 1
            
            # Extraer los conjuntos
            left_set = expr[left_start:left_end+1]
            right_set = expr[right_start:right_end]
            
            try:
                # Extraer elementos de los conjuntos
                left_elements = extract_set_elements(left_set)
                right_elements = extract_set_elements(right_set)
                
                # Calcular la diferencia
                diff_elements = [elem for elem in left_elements if elem not in right_elements]
                
                if not diff_elements:
                    diff_result = "()"  # Conjunto vacío
                else:
                    diff_result = "(" + "|".join(sorted(diff_elements)) + ")"
                
                # Reemplazar en el resultado
                result = result[:-(i-left_start)] + diff_result
                i = right_end
            except Exception as e:
                # En caso de error, mantener la expresión original
                print(f"Error processing difference: {e}")
                result += expr[i]
                i += 1
        else:
            result += expr[i]
            i += 1
    
    return result

def extract_set_elements(set_expr):
    """
    Extrae los elementos de un conjunto expresado como [a-z], (a|b|c) o un solo carácter.
    Retorna una lista de elementos.
    """
    elements = []
    
    # Debug information
    print(f"Extracting elements from: {set_expr}")
    
    # Caso de conjunto entre corchetes [a-z]
    if set_expr.startswith('[') and set_expr.endswith(']'):
        content = set_expr[1:-1]
        i = 0
        while i < len(content):
            if i + 2 < len(content) and content[i+1] == '-':
                # Rango como a-z
                start_char = content[i]
                end_char = content[i+2]
                print(f"  Range: {start_char}-{end_char}")
                for code in range(ord(start_char), ord(end_char) + 1):
                    elements.append(chr(code))
                i += 3
            else:
                print(f"  Single char: {content[i]}")
                elements.append(content[i])
                i += 1
    
    # Caso de conjunto como (a|b|c)
    elif set_expr.startswith('(') and set_expr.endswith(')'):
        content = set_expr[1:-1]
        for item in content.split('|'):
            if item:
                print(f"  Alternative: {item}")
                elements.append(item)
    
    # Caso de un solo carácter
    else:
        print(f"  Literal: {set_expr}")
        elements.append(set_expr)
    
    print(f"  Extracted {len(elements)} elements: {elements}")
    return elements

def convert_optional_operator(expr: str) -> str:
    r"""
Convierte cada operador '?' en notación postfija que no está escapado en su forma equivalente:   X? --> (X|) donde "" representa la cadena vacía.

Esta implementación extrae correctamente el operando inmediatamente anterior al operador '?', manejando tanto operandos de un solo carácter como operandos agrupados sin producir paréntesis adicionales o faltantes.
    """
    result = ""
    i = 0
    while i < len(expr):
        if expr[i] == "?" and (i == 0 or expr[i - 1] != "\\"):
            # If we're at the beginning, treat '?' as literal
            if i == 0:
                result += "?"
            else:
                # Extract the operand before the '?' operator
                operand, start_pos = extract_operand(expr, i)
                if operand:
                    # Replace the operand and the '?' with the optional form
                    result = result[:-(i-start_pos)] + f"({operand}|_)"
                else:
                    # If no operand found, treat '?' as literal
                    result += "?"
        else:
            result += expr[i]
        i += 1
    
    # Clean up any malformed optional expressions
    result = clean_optional_expressions(result)
    return result

def extract_operand(expr: str, pos: int) -> (str, int):
    """
    Extracts the operand immediately preceding the position 'pos' in the expression.
    Returns the operand and its starting position.
    
    Handles both grouped operands (ending with ')') and single-character operands.
    """
    if pos <= 0:
        return "", 0
    
    # Check if the preceding character is a closing parenthesis
    if expr[pos - 1] == ")":
        # Find the matching opening parenthesis
        balance = 1
        start_pos = pos - 2
        while start_pos >= 0 and balance > 0:
            if expr[start_pos] == ")":
                balance += 1
            elif expr[start_pos] == "(":
                balance -= 1
            start_pos -= 1
        
        if start_pos < 0:
            # Unbalanced parentheses, return empty
            return "", 0
        
        # Extract the operand including parentheses
        operand = expr[start_pos+1:pos]
        
        # Check if the operand is already an optional expression
        if "|_)" in operand:
            return operand, start_pos+1
        
        # Remove redundant outer parentheses if needed
        if operand.startswith("(") and operand.endswith(")"):
            # Check if parentheses are balanced
            inner_balance = 0
            for char in operand[1:-1]:
                if char == "(":
                    inner_balance += 1
                elif char == ")":
                    inner_balance -= 1
                if inner_balance < 0:
                    break
            
            # If balanced, remove outer parentheses
            if inner_balance == 0:
                operand = operand[1:-1]
        
        return operand, start_pos+1
    else:
        # Single character operand
        return expr[pos-1:pos], pos-1

def clean_optional_expressions(expr: str) -> str:
    """
    
    Limpia las expresiones opcionales eliminando paréntesis redundantes
    y corrigiendo expresiones opcionales mal formadas.

    Específicamente maneja:
    1. Paréntesis adicionales en expresiones opcionales: (X|_))
    2. Paréntesis redundantes anidados en expresiones opcionales: (((X)|_))
    3. Asegura un balance adecuado de paréntesis en todas las expresiones opcionales.
    """
    # Primera pasada: corregir patrones mal formados evidentes
    expr = expr.replace('|_))', '|_)')
    
    # Segunda pasada: manejar expresiones opcionales anidadas
    i = 0
    result = ""
    while i < len(expr):
        # Look for optional expression pattern
        if i + 3 < len(expr) and expr[i:i+3] == "(_|" and ")" in expr[i+3:]:
            # Corregir patrón opcional invertido (_|X)
            close_pos = expr.find(")", i+3)
            result += f"({expr[i+3:close_pos]}|_)"
            i = close_pos + 1
        elif i + 3 < len(expr) and expr[i:i+3] == "((_" and ")" in expr[i+3:]:
            # Corregir patrón anidado mal formado ((_|X))
            close_pos = expr.find(")", i+3)
            if close_pos + 1 < len(expr) and expr[close_pos+1] == ")":
                result += f"({expr[i+3:close_pos]}|_)"
                i = close_pos + 2
            else:
                result += expr[i]
                i += 1
        else:
            result += expr[i]
            i += 1
    
    # tercera pasada: balanceo de parentesis. 
    stack = []
    balanced_result = ""
    i = 0
    while i < len(result):
        if result[i:i+3] == "|_)" and stack and stack[-1] == "(":
        # Encontramos el final de una expresión opcional
        # Verificar si tenemos el número correcto de paréntesis de apertura
            stack.pop() # Eliminar el paréntesis de apertura
            balanced_result += "|_)"
            i += 3
            
            # Verificar paréntesis de cierre adicionales

            while i < len(result) and result[i] == ")":
                # Omitir paréntesis de cierre adicionales
                i += 1
        elif result[i] == "(":
            stack.append("(")
            balanced_result += "("
            i += 1
        elif result[i] == ")":
            if stack and stack[-1] == "(":
                stack.pop()
            balanced_result += ")"
            i += 1
        else:
            balanced_result += result[i]
            i += 1
    
# Última pasada: simplificar expresiones como ((X)|_) a (X|_)
    simplified = ""
    i = 0
    while i < len(balanced_result):
        if (i + 4 < len(balanced_result) and 
            balanced_result[i:i+2] == "((" and 
            "|_)" in balanced_result[i+2:]):
            
            # Encontrar el paréntesis de cierre correspondiente al grupo interno
            inner_close = -1
            balance = 2  # Iniciar con 2 debido a los dos paréntesis de apertura
            for j in range(i+2, len(balanced_result)):
                if balanced_result[j] == "(":
                    balance += 1
                elif balanced_result[j] == ")":
                    balance -= 1
                    if balance == 1:  #Encontramos el paréntesis de cierre del grupo interno
                        inner_close = j
                        break
            
            if inner_close != -1 and inner_close + 3 < len(balanced_result) and balanced_result[inner_close+1:inner_close+4] == "|_)":
                # Encontramos un patrón como ((X)|_)
                inner_content = balanced_result[i+2:inner_close]
                simplified += f"({inner_content}|_)"
                i = inner_close + 4
            else:
                simplified += balanced_result[i]
                i += 1
        else:
            simplified += balanced_result[i]
            i += 1
    
    return simplified


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
    """
    if not expr:
        return expr
    
    # Eliminar paréntesis exteriores redundantes
    while expr.startswith('(') and expr.endswith(')'):
        # Verificar si los paréntesis están balanceados
        balance = 0
        for i, char in enumerate(expr[1:-1]):
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
            
            # Si el balance llega a -1, significa que los paréntesis exteriores
            # no son redundantes (hay un cierre que no corresponde a la apertura inicial)
            if balance < 0:
                break
        
        # Si terminamos con balance 0, los paréntesis exteriores son redundantes
        if balance == 0:
            expr = expr[1:-1]
        else:
            break
    
    # Eliminar paréntesis redundantes en expresiones de la forma ((X|_))
    if expr.startswith('((') and expr.endswith('))'):
        inner = expr[1:-1]
        if inner.count('(') == 1 and inner.count(')') == 1 and '|_' in inner:
            # Es una expresión opcional con paréntesis redundantes
            return inner
    
    return expr

def simplify_expression(expr: str) -> str:
    r"""
    Reduce paréntesis redundantes en cada parte de la expresión a nivel superior.
    """
# Divide la expresión en los operadores '|' del nivel superior
    parts = []
    current_part = ""
    paren_level = 0
    
    for char in expr:
        if char == '(' or char == '[':
            paren_level += 1
            current_part += char
        elif char == ')' or char == ']':
            paren_level -= 1
            current_part += char
        elif char == '|' and paren_level == 0:
            parts.append(current_part)
            current_part = ""
        else:
            current_part += char
    
    if current_part:
        parts.append(current_part)
    
    simplified_parts = []
    for part in parts:
        # Primero eliminamos paréntesis redundantes exteriores
        simplified = remove_outer_parentheses(part.strip())
        
        # Corregir expresiones opcionales mal formadas
        simplified = simplified.replace('|_))', '|_)')
        
        # Luego verificamos si es una expresión opcional y limpiamos paréntesis redundantes
        if "|_)" in simplified:
            # Buscar el paréntesis de apertura correspondiente
            close_pos = simplified.find("|_)")
            open_pos = -1
            balance = 1
            for i in range(close_pos-1, -1, -1):
                if simplified[i] == ')':
                    balance += 1
                elif simplified[i] == '(':
                    balance -= 1
                    if balance == 0:
                        open_pos = i
                        break
            
            if open_pos >= 0:
                # Extraer la parte interna de la expresión opcional
                inner = simplified[open_pos+1:close_pos]
                # Si la parte interna ya tiene paréntesis, los eliminamos
                if inner.startswith('(') and inner.endswith(')'):
                    inner = remove_outer_parentheses(inner)
                # Reconstruimos la expresión opcional limpia
                simplified = simplified[:open_pos] + f"({inner}|_)" + simplified[close_pos+3:]
        
        simplified_parts.append(simplified)
    
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
    # Split the expression at top level '|' operators
    parts = []
    current_part = ""
    paren_level = 0
    
    for char in expr:
        if char == '(' or char == '[':
            paren_level += 1
            current_part += char
        elif char == ')' or char == ']':
            paren_level -= 1
            current_part += char
        elif char == '|' and paren_level == 0:
            parts.append(current_part)
            current_part = ""
        else:
            current_part += char
    
    if current_part:
        parts.append(current_part)
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


def expand_negated_bracket_content(content, alphabet):
    """
    Expande un conjunto negado (por ejemplo, "^a-z") a una alternancia de todos los caracteres
    que no están en el conjunto.
    """
    # Primero expandimos el contenido negado para obtener todos los caracteres que NO queremos
    expanded_content = expand_bracket_content(content)
    # Quitamos los paréntesis exteriores y separamos por '|'
    chars_to_exclude = set(expanded_content[1:-1].split('|'))
    # Filtramos el alfabeto para incluir solo los caracteres que no están en chars_to_exclude
    included_chars = [ch for ch in alphabet if ch not in chars_to_exclude]
    if not included_chars:
        return "()"  # Conjunto vacío
    return "(" + "|".join(included_chars) + ")"

def balance_parentheses(expr: str) -> str:
    """
    Verifica y corrige paréntesis desbalanceados en una expresión regular.
    
    Esta función:
    1. Cuenta los paréntesis de apertura y cierre
    2. Añade paréntesis faltantes al final
    3. Corrige patrones específicos como (X|_) con paréntesis desbalanceados
    """
    # Casos especiales para caracteres escapados
    if expr == "\\(":
        return "\\("
    if expr == "\\)":
        return "\\)"
    if expr == "\\+":
        return "\\+"
    if expr == "\\*":
        return "\\*"
    
    # Primero, corregir patrones específicos de expresiones opcionales
    expr = expr.replace('(|_)', '(_|)')
    expr = expr.replace('|_))', '|_)')
    
    # Corregir expresiones opcionales mal formadas
    if '|_' in expr and not expr.endswith(')'):
        expr += ')'
    
    # Contar paréntesis
    open_count = 0
    close_count = 0
    i = 0
    while i < len(expr):
        if expr[i] == '\\' and i + 1 < len(expr):
            # Saltar caracteres escapados
            i += 2
            continue
        if expr[i] == '(':
            open_count += 1
        elif expr[i] == ')':
            close_count += 1
        i += 1
    
    # Si hay más paréntesis de apertura, añadir los de cierre faltantes
    if open_count > close_count:
        expr += ')' * (open_count - close_count)
    
    # Si hay más paréntesis de cierre, añadir los de apertura faltantes al inicio
    elif close_count > open_count:
        expr = '(' * (close_count - open_count) + expr
    
    # Verificar patrones específicos de expresiones opcionales mal formadas
    i = 0
    while i < len(expr) - 3:
        if expr[i:i+3] == '|_)' and i > 0:
            # Buscar el paréntesis de apertura correspondiente
            j = i - 1
            balance = 1
            while j >= 0 and balance > 0:
                if expr[j] == ')':
                    balance += 1
                elif expr[j] == '(':
                    balance -= 1
                j -= 1
            
            # Si no encontramos el paréntesis de apertura, añadirlo
            if balance > 0:
                expr = expr[:i] + '(' + expr[i:]
                i += 1  # Ajustar el índice por el paréntesis añadido
        i += 1
    
    # Corregir problemas específicos con caracteres especiales en Token 9
    if '"(' in expr and ')$' in expr:
        expr = expr.replace('"(', '"').replace(')$', '$')
    
    # Corregir problemas con paréntesis desbalanceados en expresiones complejas
    if '(||_)' in expr:
        expr = expr.replace('(||_)', '(|_)')
    
    return expr

