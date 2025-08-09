# tests/test_ad_batch.py
from typing import List, Tuple

import pytest

from src.config import TASK_CONFIGS
from src.predict import predict_batch

# (문장, 기대 라벨)
CASES: List[Tuple[str, str]] = [
    # 광고
    ("최신폰 무료 증정 이벤트!", "광고"),
    ("지금 클릭하면 무료 증정!", "광고"),
    ("지금 구매하시면 사은품 드려요", "광고"),
    ("단 24시간! 1+1 특가", "광고"),
    ("선착순 100명 쿠폰 지급", "광고"),
    ("링크 접속 시 즉시 할인", "광고"),
    ("이 링크로 신청하세요", "광고"),
    ("체험단 모집! 지원 링크", "광고"),
    ("배송비 무료 + 추가 10% 쿠폰", "광고"),
    ("한정 수량 마지막 기회", "광고"),
    ("구독하면 포인트 5천원", "광고"),
    ("스폰서 협찬 안내드립니다", "광고"),
    ("제휴 문의는 DM 주세요", "광고"),
    ("0원 설치, 상담 예약 클릭", "광고"),
    ("겨울 세일 최대 70% OFF", "광고"),

    # 정상
    ("반가워요", "정상"),
    ("기분이 좋더라구요~", "정상"),
    ("오늘 날씨 참 좋네요.", "정상"),
    ("아이잘때 달아놓는 캠을 통해 우연히 시어머니가 맨발로 침대위에서 발로 앉으시고 발을 쫙 핀상태로 누으시는 장면을 보게됬고 그걸 남편한테 자연스럽게 말을 했어요.. ", "정상"),
    ("남편도 그건 잘못된거라 했지만 아이를 봐주시러 와주시는 시어머니에게 않좋게 말씀드리고 싶지는 않은데 어떡하면 심기를 불편하게 해드리지 않고 자연스러운 방법이 없을까요? ", "정상"),
    ("점심 뭐 드셨어요?", "정상"),
    ("이번 주말에 가족 모임이 있어요.", "정상"),
    ("배초보고 7,8알 숙제내주셨는데… 늦어지는걸까요? 아님 지나간 걸까요.. 아님 배란이 안된걸까요…?", "정상"),
    ("이렇게 단호박이어도 되나요 ㅠㅠㅠㅠㅋㅋㅋ 태어나서 본 줄 중에 가장 선명한 한줄이네요  ㅠㅠㅠ 이렇게 안선명해도 된단다^^😭 배란10일째여서 해봤는데 ㅠㅠ 너무 단단한 단호박🤣 나쁜넘 ㅠㅠㅠㅠㅠ😂 기대한만큼 실망도 커지네옹 ", "정상"),
    ("어디다 물어볼 수도 없고 저는 남편이랑 사랑 나누고 나면 소중이가 아니라 목이 아픕니다", "정상"),
    ("혹시 원포 두줄 초초초초초초촟초매직아이로 보이시나요?? 근데 만약 맞다하더라도 저 위치에 줄이 나타나진 않죠,,?ㅎㅎ", "정상"),
    ("단호박 한줄이죠,,?", "정상"),
    ("4년째 자임시도중인데 얼마나 예쁜천사가 오려고 이렇게 늦게 올까요ㅠㅠ 이번년도엔 꼭 천사보길 원하면서 퇴사하고 매일 운동중이네용 다들 저처럼 오래 자임이 안되는걸까요?ㅠㅠ", "정상"),
    ("7월 25일 ~ 31일 생리 어플상 8월 9일이 배란일 예상인데 그럼 이번달은 피크없이 지나가는건가요.? ㅜㅜ 배란에 문제가 있는거겠죠 ? 배테기 이용 초보라 도움 부탁드려요.", "정상"),
    ("쿠팡으로 시키려는데 어떤게 좋은가요?ㅠㅠ", "정상"),
    ("여건상 제가 거실에서 업무를 봐야할경우는 시어머니가 그 시간에 맞춰 아이를 봐주러 오시는데 다른방에는 에어컨이 따로 없는관계로 안방 부부침대에서 아이를 2시간? 정도 봐줘야 하는경우가 일주일에 한두번 정도 있어요~ ", "정상"),
    ("배란 10일차인데ㅠㅠㅠㅠ 눈이만든 가짜선인지…두줄이라해도 10일차에 이정도 연하기이면 아닌거겠죠?? ", "정상"),
]

IDS = [f"{i:02d}_{'ad' if lab == '광고' else 'normal'}" for i, (_, lab) in enumerate(CASES)]

# 스모크: 고정 선택
SMOKE_IDX = [0, 1]
SMOKE_CASES = [CASES[i] for i in SMOKE_IDX]
SMOKE_IDS = [IDS[i] for i in SMOKE_IDX]


@pytest.mark.parametrize("text,expected", SMOKE_CASES, ids=SMOKE_IDS)
def test_ad_batch_predictions(model_and_tokenizer, max_len, device, text, expected):
    """
    배치 경로를 사용해 단건도 예측해 동일 코드 경로를 검증.
    스모크 케이스만 사용해 실패 중복 출력을 방지.
    """
    model, tokenizer = model_and_tokenizer
    preds = predict_batch(
        model=model,
        tokenizer=tokenizer,
        sentences=[text],
        max_len=max_len,
        device=device,
        label2str=TASK_CONFIGS["ad"]["label2str"],
        log_stats=False,
    )
    assert len(preds) == 1, "예측 결과가 1개가 아닙니다."
    got = preds[0]
    assert got == expected, f"입력: {text}\n예측: {got}\n기대: {expected}"


def test_ad_batch_predictions_all(model_and_tokenizer, max_len, device):
    """
    모든 케이스를 배치 예측으로 검증하고
    틀린 항목이 있으면 간단하고 직관적으로 출력.
    """
    model, tokenizer = model_and_tokenizer
    sentences = [t for t, _ in CASES]
    expected = [e for _, e in CASES]

    preds = predict_batch(
        model=model,
        tokenizer=tokenizer,
        sentences=sentences,
        max_len=max_len,
        device=device,
        label2str=TASK_CONFIGS["ad"]["label2str"],
        log_stats=False,
    )

    mismatches = [
        (i, inp, exp, got)
        for i, (inp, exp, got) in enumerate(zip(sentences, expected, preds))
        if exp != got
    ]

    if mismatches:
        lines = ["예측 불일치 목록:"]
        for idx, inp, exp, got in mismatches:
            lines.append(f"[{idx}] {inp}  (기대: {exp} → 예측: {got})")
        raise AssertionError("\n".join(lines))

    assert preds == expected
