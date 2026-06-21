"""
영어 → 한글 표시용 매핑 (공유 모듈)

원칙: 데이터 원본·모델 내부는 영어 유지. 화면/그래프 '표시'에만 한글 입힘.
사용: from common.labels_ko import CROP_KO, FEATURE_KO, ko_crop, ko_feature
사유: ml·dl·llm·Streamlit이 다 import해서 재사용
"""

# 작물 22종 (Crop Recommendation 데이터 label)
CROP_KO = {
    "rice": "벼(쌀)",
    "maize": "옥수수",
    "chickpea": "병아리콩",
    "kidneybeans": "강낭콩",
    "pigeonpeas": "비둘기콩",
    "mothbeans": "모스빈",
    "mungbean": "녹두",
    "blackgram": "검은녹두",
    "lentil": "렌틸콩",
    "pomegranate": "석류",
    "banana": "바나나",
    "mango": "망고",
    "grapes": "포도",
    "watermelon": "수박",
    "muskmelon": "머스크멜론",
    "apple": "사과",
    "orange": "오렌지",
    "papaya": "파파야",
    "coconut": "코코넛",
    "cotton": "목화",
    "jute": "황마",
    "coffee": "커피",
}

# 피처 7종
FEATURE_KO = {
    "N": "질소(N)",
    "P": "인(P)",
    "K": "칼륨(K)",
    "temperature": "온도(℃)",
    "humidity": "습도(%)",
    "ph": "산도(pH)",
    "rainfall": "강수량(mm)",
}


def ko_crop(name: str) -> str:
    """작물 영어명 → 한글 (없으면 원문)"""
    return CROP_KO.get(name, name)


def ko_feature(name: str) -> str:
    """피처 영어명 → 한글 (없으면 원문)"""
    return FEATURE_KO.get(name, name)
