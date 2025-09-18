from openai import OpenAI

class PersianNER:
    def __init__(self, api_key: str, base_url: str = "https://api.metisai.ir/openai/v1"):
        """
        Initialize the NER extractor with API key and optional base_url.
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # Base instruction for the model
        self.system_prompt = (
            "شما یک مدل هوش مصنوعی هستید که وظیفه استخراج موجودیت‌ها (Entity) "
            "از متن فارسی را دارد."
        )

    def extract_entities(self, text: str, model: str = "gpt-4.1-mini") -> list[str]:
        """
        Extract named entities from Persian text.
        Returns a list of strings (entities).
        """
        user_prompt = (
            "متن زیر را بخوان و تمام موجودیت‌ها (Entity) مانند نام افراد، مکان‌ها، "
            "سازمان‌ها، تاریخ‌ها، رویدادها، آثار، و اصطلاحات خاص را به‌صورت یک لیست استخراج کن. "
            "نیازی به توضیح اضافه یا بازنویسی متن نیست. فقط لیست موجودیت‌ها را ارائه بده.\n\n"
            f"متن:\n{text}"
        )

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        # Extract model's reply
        raw_output = response.choices[0].message.content

        # Try to split entities into list (assuming model outputs line-separated or comma-separated items)
        entities = [e.strip("•-– \n") for e in raw_output.replace("\n", ",").split(",") if e.strip()]
        
        return entities