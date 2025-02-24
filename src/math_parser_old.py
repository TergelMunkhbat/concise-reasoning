import re
import ast
import regex

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", string)
    _string = re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", _string)
    return _string


def _fix_tan(string):
    _string = re.sub(r"\\tan(-?[0-9.a-zA-Z]+)", r"\\tan{\1}", string)
    _string = re.sub(r"\\tan\s+(\w+)$", r"\\tan{\1}", _string)
    return _string


def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")

    # replace \\ with \
    # string = string.replace("\\\\", "\\")
    # string = string.replace("\\\\", "\\")

    if string.startswith("\\text{") and string.endswith("}"):
        string = string.split("{", 1)[1][:-1]

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("cfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "").strip()
    string = string.replace("^\\circ", "").strip()

    string = regex.sub(r"\{(c|m)?m\}(\^(2|3))?", "", string).strip()
    string = regex.sub(r"p\.m\.$", "", string).strip()
    string = regex.sub(r"(\d)\s*t$", r"\1", string).strip()

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "%")
    string = string.replace("\%", "%")
    # string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and 
    # string = string.replace("and", "")
    string = string.replace("\\mathbf", "")
    string = string.replace("\\mathrm", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace("\"", "")
    
    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    # if len(string.split("=")) == 2:
    #     if len(string.split("=")[0]) <= 2:
    #         string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = _fix_tan(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    string = regex.sub(r"(\\|,|\.)+$", "", string)

    return string

def extract_boxed_answers(text):
    answers = []
    for piece in text.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    return answers


def parse_fraction(input_str):
    """
    Parses LaTeX-like fraction strings (including negatives) and evaluates them as float decimals.
    If execution exceeds the timeout limit, it raises an error.
    """
    cleaned_str = input_str.replace("\\dfrac", "").replace("\\frac", "")
    
    is_negative = cleaned_str.startswith("-")
    
    # Strip the negative sign for easier parsing
    if is_negative:
        cleaned_str = cleaned_str[1:]
    
    cleaned_str_lst = cleaned_str.strip("{}").split("}{")
    if len(cleaned_str_lst) != 2:
        return None
    else:
        numerator = cleaned_str_lst[0]
        denominator = cleaned_str_lst[1]
    if re.fullmatch(r"-?\d+", numerator):
        numerator = int(numerator)
    else:
        return None
    if re.fullmatch(r"-?\d+", denominator):
        denominator = int(denominator)
        if denominator == 0:
            return None
    else:
        return None
    
    result = numerator / denominator
    
    if is_negative:
        result = -result
    
    return str(float(result))



def extract_answer(pred_str, exhaust=False):
    pred = []

    if 'boxed' in pred_str:
        pred = extract_boxed_answers(pred_str)
    elif ('answer is' in pred_str):
        # pred = [pred_str.split('answer is')[-1].strip()]
        pred_temp = pred_str.split('answer is')[-1].strip()
        list_temp = pred_str.split('answer is')[-1].strip().split()
        # get rid of units: if first element is number, and second number is alphabet, only get the first number
        if len(list_temp) > 1:
            if list_temp[0].replace(",", "").replace(".", "").isdigit() and list_temp[1][0].isalpha():
                # print(list_temp, list_temp[0])
                pred.append(list_temp[0])
            else:
                pred.append(pred_temp)
        else:
            pred.append(pred_temp)
    else: # use the last number
        pattern = '-?\d*\.?\d+'
        ans = re.findall(pattern, pred_str.replace(",", ""))
        if(len(ans) >= 1):
            ans = ans[-1]
        else:
            ans = ''
        if ans:
            pred.append(ans)

    # multiple line
    _pred = []
    for ans in pred:
        # ans = ans.strip().split("\n")[0]
        ans = ans.strip()
        
        ans = ans.lstrip(":")
        ans = ans.rstrip(".")
        ans = ans.rstrip("/")
        ans = strip_string(ans)
        _pred.append(ans)

    if exhaust:
        return _pred
    else:
        return _pred[-1] if _pred else ""


def extract_math_answer(question, reasoning):
    answer = []
    # handle cases where desired answer is multiple separated by commas => return list of lists
    if 'separated by commas' in question:
        for ans in extract_answer(reasoning, exhaust=True):
            ans = ans.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
            temp = [a.strip() for a in ans.split(",")]
            answer.append(temp)
            # handle fractions
            num_temp = []
            frac_flag = False
            for a in temp:
                if 'frac' in a:
                    frac_a = parse_fraction(a)
                    if frac_a is not None:
                        num_temp.append(frac_a)
                        frac_flag = True
                    else:
                        num_temp.append(a)
                else:
                    num_temp.append(a)
            if frac_flag:
                answer.append(num_temp)
        return answer
    # handle cases where desired answer is one (but could be multiple) => return list of strings
    else:
        for ans in extract_answer(reasoning, exhaust=True):
            if regex.search(r"\\text\{\s*and\s*\}", ans):
                temp = [a.strip() for a in regex.sub(r"\\text\{\s*and\s*\}", "[SEP]", ans).split("[SEP]")]
                for a in temp:
                    if 'frac' in a:
                        frac_a = parse_fraction(a)
                        if frac_a is not None:
                            temp.append(frac_a)
                answer.extend(temp)
            else:
                answer.append(ans.strip())
                if 'frac' in ans.strip():
                    frac_a = parse_fraction(ans.strip())
                    if frac_a is not None:
                        answer.append(frac_a)
        return list(set(answer))


def compare_answers(question, parsed_label, parsed_answer):
    
    # Convert to lists if they're string representations of lists
    if isinstance(parsed_label, str):
        parsed_label = ast.literal_eval(parsed_label)
    if isinstance(parsed_answer, str):
        parsed_answer = ast.literal_eval(parsed_answer)
    
    if 'separated by commas' in question:
        # Convert each sublist to a frozen set for hashability
        answer_sets = {frozenset(str(x) for x in ans) for ans in parsed_answer}
        label_sets = {frozenset(str(x) for x in lab) for lab in parsed_label}
        # Check for any intersection between answer_sets and label_sets
        return bool(answer_sets & label_sets)
    else:
        # Convert lists to sets for efficient membership testing
        return bool(set(str(x) for x in parsed_answer) & set(str(x) for x in parsed_label))


if __name__ == "__main__":
    import json

    def process_file(filename):
        with open(filename, 'r') as file:
            return [json.loads(line) for line in file if line.strip()]

    file_name = "output.json"
    output_data = process_file(file_name)

    for data in output_data:
        data['parsed_label'] = extract_math_answer(data['input'], data['solution'])
    
    for data in output_data:
        data['parsed_answer'] = extract_math_answer(data['input'], data['rationale'])

    for data in output_data:
        data['correct'] = compare_answers(data['input'], data['parsed_label'], data['parsed_answer'])

    count_correct = 0
    for data in output_data:
        if data['correct']:
            count_correct += 1

    print(f"Correct: {count_correct}")
    print(f"Accuracy: {count_correct/len(output_data)*100:.2f}")

