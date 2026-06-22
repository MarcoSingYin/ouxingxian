import os
import json
import pdfplumber
from openai import OpenAI

# ---------------- 配置区（仅保留 PDF 路径与输出文件名） ----------------
PDF_PATH = "A case of portal vein recanalization and symptomatic heart failure.pdf"
OUTPUT_JSON = "提取病例实体.json"

# 为防止超长文本超过模型上下文限制
MAX_INPUT_LENGTH = 100000
# --------------------------------------------------------------------


# ==================================================
# PDF 读取
# ==================================================
def extract_pdf_text(pdf_path):
    """
    读取 PDF 内容并合并所有页的文本
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在：{pdf_path}")

    text_list = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_list.append(page_text.strip())

    return "\n".join(text_list)


# ==================================================
# Prompt 构造
# ==================================================
def build_prompt(text):
    """
    构建用于信息抽取的指令，并嵌入论文内容
    """
    # 超长文本截断
    text = text[:MAX_INPUT_LENGTH]

    prompt = f"""
    你是一个专业的医学文本信息抽取助手。请从以下论文内容中提取病例实体信息，严格按照下面的 JSON 结构输出。要求：

    1. **不预定义任何具体症状、疾病或药物名称**，所有字段内容必须从原文中直接提取或概括。
    2. 对于“主要症状”，请将原文中描述的患者不适、异常表现等，以数组形式列出。每个症状元素包含：
       - “症状名称”：简明概括该症状（如“消化道出血”“呼吸困难”“发热”等）
       - “详细描述”：原文中的具体表现（颜色、量、频率、诱因、缓解因素等）
    3. 如果原文没有某字段的信息，保留空字符串 `""` 或空数组 `[]`，但不要删除该字段。
    4. 输出必须是**合法 JSON**，不要有任何额外解释、注释或 markdown 标记。

    【输出 JSON 结构】
    {{
      "患者基本信息": {{
        "年龄": "",
        "性别": "",
        "其他": ""
      }},
      "主要症状": [
        {{
          "症状名称": "",
          "详细描述": ""
        }}
      ],
      "既往史": {{
        "既往疾病史": "",
        "用药史": "",
        "手术史": "",
        "过敏史": "",
        "其他": ""
      }},
      "诊断结果": {{
        "主要诊断": "",
        "并发症": "",
        "实验室检查异常": "",
        "影像学发现": ""
      }},
      "治疗方案": {{
        "药物治疗": "",
        "介入治疗": "",
        "手术治疗": "",
        "其他支持治疗": ""
      }}
    }}

    【填充细则】
    - 患者基本信息中的“其他”：用于填写非年龄/性别的特征（如“高龄”“儿童”等）。
    - 主要症状：每条独立症状请单独列出一个数组元素。例如：“反复便血，每日2-3次，每次100-200g” → 症状名称：“便血”，详细描述：“反复便血，每日2-3次，每次100-200g，鲜红色”。
    - 既往史：只记录本次发病前的疾病、用药、手术等。用药史写明药物名称、用法（若提供）。
    - 诊断结果：包括最终诊断名称、重要并发症、关键异常检验值（如BNP 1087 ng/L）、影像学结论。
    - 治疗方案：区分口服药、静脉药、介入操作、手术、输血/营养支持等。若原文未区分，可合并写入最合适的子类。

    【论文内容】
    {text}
    """

    return prompt


# ==================================================
# 调用大模型（需传入用户提供的 API 信息）
# ==================================================
def extract_info_with_llm(text, api_key, base_url, model):
    """
    使用用户提供的 API 凭据和模型，向 LLM 请求结构化抽取结果
    """
    prompt = build_prompt(text)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你是专业医疗病历结构化抽取专家，只返回JSON。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1
        )
        return completion.choices[0].message.content

    except Exception as e:
        print(f"[ERROR] API调用失败：{e}")
        return None


# ==================================================
# 清洗模型返回内容（去除可能的 markdown 标记）
# ==================================================
def clean_json_response(response_text):
    if not response_text:
        return ""

    response_text = response_text.strip()

    if response_text.startswith("```json"):
        response_text = response_text.replace("```json", "", 1)
    if response_text.startswith("```"):
        response_text = response_text.replace("```", "", 1)
    if response_text.endswith("```"):
        response_text = response_text[:-3]

    return response_text.strip()


# ==================================================
# JSON 解析与校验
# ==================================================
def parse_json(response_text):
    try:
        cleaned_text = clean_json_response(response_text)
        result = json.loads(cleaned_text)
        if not isinstance(result, dict):
            raise ValueError("返回结果不是JSON对象")
        return result
    except Exception as e:
        print(f"[ERROR] JSON解析失败：{e}")
        print("\n===== 模型返回内容 =====\n")
        print(response_text)
        return None


# ==================================================
# 保存 JSON 文件
# ==================================================
def save_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ==================================================
# 主程序
# ==================================================
def main():
    print("=" * 50)
    print("  医疗病例实体抽取工具")
    print("=" * 50)

    # 由用户交互输入敏感信息
    print("\n请配置 API 连接信息（以下信息不会保存，仅本次运行有效）：")
    api_key = input("请输入您的 API Key: ").strip()
    base_url = input("请输入 API 调用链接（例如 https://dashscope.aliyuncs.com/compatible-mode/v1）: ").strip()
    model = input("请输入选择的模型名称（例如 qwen-turbo）: ").strip()

    # 简单校验
    if not api_key or not base_url or not model:
        print("[ERROR] 所有配置项均不能为空，程序退出。")
        return

    print("\n[INFO] 开始读取 PDF...")
    try:
        pdf_text = extract_pdf_text(PDF_PATH)
        if not pdf_text:
            print("[ERROR] PDF 内容为空")
            return
        print(f"[INFO] 成功读取 {len(pdf_text)} 字符")
    except Exception as e:
        print(f"[ERROR] PDF 读取失败：{e}")
        return

    print("[INFO] 调用大模型进行实体抽取（可能需要一段时间）...")
    result_text = extract_info_with_llm(pdf_text, api_key, base_url, model)

    if not result_text:
        print("[ERROR] API 返回为空")
        return

    result = parse_json(result_text)
    if not result:
        return

    try:
        save_json(result, OUTPUT_JSON)
        print(f"\n[SUCCESS] 提取完成！")
        print(f"[SUCCESS] JSON 已保存至：{OUTPUT_JSON}")
    except Exception as e:
        print(f"[ERROR] JSON 保存失败：{e}")


if __name__ == "__main__":
    main()