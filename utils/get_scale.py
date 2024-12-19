import re


# Expressão regular para extrair o valor de ratio
ratio_pattern = r"scale (\d+(\.\d+)?)"

def get_scale(filename):
    # Procurar pelo padrão na string
    match = re.search(ratio_pattern, filename)

    # Se encontrar, imprimir o valor de ratio
    if match:
        return float(match.group(1))
        # print(f"Valor de ratio: {ratio_value}")
    else:
        # print("Valor de ratio não encontrado")
        return None


if __name__ == "__main__":

    # Nome do arquivo
    file_names = [
        "ratio 4.5 seed 0.png",
        "scale 0.1 ratio 4.5 seed 2",
    ]
    for filename in file_names:
        scale = get_scale(filename)
        print(scale, type(scale))

