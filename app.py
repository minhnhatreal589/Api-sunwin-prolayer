import requests
from flask import Flask, jsonify
from collections import Counter, defaultdict
import math
import os
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# URL API gốc để lấy dữ liệu
SOURCE_API_URL = "https://ahihidonguoccut-2b5i.onrender.com/mohobomaycai"

# Biến toàn cục lưu trữ lịch sử các phiên và dự đoán (lưu full, không giới hạn)
historical_data = []
prediction_history = []  # Lưu lịch sử dự đoán: {"session": int, "result": str, "du_doan": str, "danh_gia": str}

# --- CÁC MODEL AI NÂNG CẤP ---
# Mỗi model trả về: (prediction, confidence, reason)

def model_probability(pattern: str, window_sizes=[10, 20, 50]):
    """Thống kê xác suất với rolling window"""
    if not pattern:
        return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < min(window_sizes):
        return 'T', 0.5, f"Chưa đủ {min(window_sizes)} phiên, mặc định Tài."
    scores = []
    for window in window_sizes:
        if len(pattern) >= window:
            recent = pattern[-window:]
            t_ratio = recent.count('T') / window
            if t_ratio > 0.7:
                scores.append(('X', t_ratio - 0.5, f"Rolling {window}: Tài cao ({t_ratio:.2f}), ưu tiên Xỉu."))
            elif t_ratio < 0.3:
                scores.append(('T', 0.5 - t_ratio, f"Rolling {window}: Xỉu cao ({t_ratio:.2f}), ưu tiên Tài."))
    if not scores:
        return None, 0.0, "Không có tín hiệu."
    vote_count = Counter([s[0] for s in scores])
    pred = vote_count.most_common(1)[0][0]
    conf = sum(s[1] for s in scores if s[0] == pred) / len([s for s in scores if s[0] == pred])
    reason = " | ".join(s[2] for s in scores if s[0] == pred)
    return pred, conf, reason

def model_markov(pattern: str, order=1):
    """Markov Chain: Xác suất chuyển trạng thái (order=1)"""
    if not pattern:
        return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 5:
        return 'T', 0.5, f"Chưa đủ 5 phiên, mặc định Tài."
    transitions = defaultdict(lambda: defaultdict(int))
    for i in range(len(pattern) - order):
        state = pattern[i:i+order]
        next_state = pattern[i+order]
        transitions[state][next_state] += 1
    current_state = pattern[-order:]
    if current_state not in transitions:
        return None, 0.0, "Không có transition."
    total = sum(transitions[current_state].values())
    probs = {k: v / total for k, v in transitions[current_state].items()}
    if not probs:
        return None, 0.0, "Không có prob."
    pred = max(probs, key=probs.get)
    conf = probs[pred] * min(1.0, len(pattern) / 50)
    reason = f"Markov (order {order}): Từ '{current_state}' -> '{pred}' với prob {probs[pred]:.2f}."
    return pred, conf, reason

def model_ngram(pattern: str, n_range=(3,6)):
    """N-gram Pattern Matching"""
    if not pattern:
        return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < max(n_range) + 1:
        return 'T', 0.5, f"Chưa đủ {max(n_range)+1} phiên, mặc định Tài."
    scores = []
    for n in range(n_range[0], n_range[1]+1):
        if len(pattern) < n:
            continue
        last_n = pattern[-n:]
        matches = [i for i in range(len(pattern) - n) if pattern[i:i+n] == last_n]
        if len(matches) < 2:
            continue
        next_chars = [pattern[i+n] for i in matches if i+n < len(pattern)]
        count = Counter(next_chars)
        if not count:
            continue
        pred, freq = count.most_common(1)[0]
        conf = (freq / len(next_chars)) * min(1.0, len(next_chars) / 5)
        scores.append((pred, conf, f"N-gram {n}: Sau '{last_n}' ra '{pred}' ({freq}/{len(next_chars)})."))
    if not scores:
        return None, 0.0, "Không có match."
    vote_count = Counter([s[0] for s in scores])
    pred = vote_count.most_common(1)[0][0]
    conf = sum(s[1] for s in scores if s[0] == pred) / len([s for s in scores if s[0] == pred])
    reason = " | ".join(s[2] for s in scores if s[0] == pred)
    return pred, conf, reason

def model_heuristic(pattern: str, sessions: list):
    """Heuristic: Kết hợp pattern cơ bản"""
    if not pattern:
        return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    sub_models = [
        model_1_bet_va_1_1(pattern),
        model_2_cau_nhip(pattern),
        model_4_cau_phuc_tap(pattern),
    ]
    valid = [m for m in sub_models if m[0] is not None]
    if not valid:
        return None, 0.0, "Không có tín hiệu."
    vote_count = Counter([m[0] for m in valid])
    pred = vote_count.most_common(1)[0][0]
    conf = sum(m[1] for m in valid if m[0] == pred) / len([m for m in valid if m[0] == pred])
    reason = "Heuristic: Kết hợp pattern cơ bản."
    return pred, conf, reason

def model_1_bet_va_1_1(pattern: str):
    """Phân tích cầu bệt và 1-1"""
    if not pattern:
        return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 3:
        return 'T', 0.5, "Chưa đủ 3 phiên, mặc định Tài."
    if pattern.endswith('TTT'):
        b_len = len(pattern) - len(pattern.rstrip('T'))
        conf = min(1.0, 0.6 + (b_len - 3) * 0.1)
        return 'T', conf, f"Bệt Tài {b_len} phiên."
    if pattern.endswith('XXX'):
        b_len = len(pattern) - len(pattern.rstrip('X'))
        conf = min(1.0, 0.6 + (b_len - 3) * 0.1)
        return 'X', conf, f"Bệt Xỉu {b_len} phiên."
    if pattern[-3:] in ["TXT", "XTX"]:
        return ('T' if pattern[-1] == 'X' else 'X'), 0.8, "Cầu 1-1."
    return None, 0.0, "Không tín hiệu."

def model_2_cau_nhip(pattern: str):
    """Phân tích cầu nhịp 1-2, 2-1, 2-2"""
    if not pattern:
        return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 4:
        return 'T', 0.5, "Chưa đủ 4 phiên, mặc định Tài."
    if pattern[-4:] in ["XXTT", "TTXX"]:
        return ('X' if pattern[-4:] == "XXTT" else 'T'), 0.85, "Cầu 2-2."
    if pattern[-3:] in ["TXX", "XTT"]:
        return ('T' if pattern[-3:] == "TXX" else 'X'), 0.75, "Cầu 1-2."
    if pattern[-3:] in ["TTX", "XXT"]:
        return ('X' if pattern[-3:] == "TTX" else 'T'), 0.75, "Cầu 2-1."
    return None, 0.0, "Không tín hiệu."

def model_3_thong_ke(pattern: str):
    """Phân tích thống kê mẫu lặp lại"""
    if not pattern:
        return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 10:
        return 'T', 0.5, "Chưa đủ 10 phiên, mặc định Tài."
    last_3 = pattern[-3:]
    starts = [i for i in range(len(pattern) - 3) if pattern[i:i+3] == last_3]
    if len(starts) < 2:
        return None, 0.0, f"Mẫu '{last_3}' ít."
    nexts = [pattern[i+3] for i in starts if i+3 < len(pattern)]
    count = Counter(nexts)
    if not count:
        return None, 0.0, "Không dữ liệu."
    pred, num = count.most_common(1)[0]
    conf = (num / len(nexts)) * min(1.0, len(nexts) / 4)
    return pred, conf, f"Sau '{last_3}' ra '{pred}' ({num}/{len(nexts)})."

def model_4_cau_phuc_tap(pattern: str):
    """Phân tích cầu 3-1, 1-3"""
    if not pattern:
        return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 5:
        return 'T', 0.5, "Chưa đủ 5 phiên, mặc định Tài."
    if pattern[-4:] == "TTTX":
        return 'X', 0.7, "Cầu 3-1."
    if pattern[-4:] == "XXXT":
        return 'T', 0.7, "Cầu 1-3."
    return None, 0.0, "Không tín hiệu."

def model_5_cau_4_1(pattern: str):
    """Phân tích cầu 4-1, 1-4"""
    if not pattern:
        return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 6:
        return 'T', 0.5, "Chưa đủ 6 phiên, mặc định Tài."
    if pattern[-5:] == "TTTTX":
        return 'X', 0.65, "Cầu 4-1."
    if pattern[-5:] == "XXXXT":
        return 'T', 0.65, "Cầu 1-4."
    return None, 0.0, "Không tín hiệu."

def model_6_cau_2_3(pattern: str):
    """Phân tích cầu 2-3, 3-2"""
    if not pattern:
        return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 6:
        return 'T', 0.5, "Chưa đủ 6 phiên, mặc định Tài."
    if pattern[-5:] == "TTXXX":
        return 'T', 0.7, "Cầu 2-3."
    if pattern[-5:] == "XXTTT":
        return 'X', 0.7, "Cầu 3-2."
    return None, 0.0, "Không tín hiệu."

def model_7_score_trend(sessions: list):
    """Phân tích xu hướng điểm số"""
    if not sessions:
        return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(sessions) < 8:
        return 'T', 0.5, "Chưa đủ 8 phiên, mặc định Tài."
    scores = [s['Tong'] for s in sessions[-8:]]
    half = len(scores) // 2
    avg1 = sum(scores[:half]) / half
    avg2 = sum(scores[half:]) / half
    diff = avg2 - avg1
    conf = min(1.0, abs(diff) / 3.0)
    if diff > 0.75:
        return 'T', conf, f"Điểm tăng ({avg2:.1f} > {avg1:.1f})."
    if diff < -0.75:
        return 'X', conf, f"Điểm giảm ({avg2:.1f} < {avg1:.1f})."
    return None, 0.0, "Xu hướng không rõ."

def model_8_cau_1_2_1(pattern: str):
    """Phân tích cầu 1-2-1"""
    if not pattern:
        return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 5:
        return 'T', 0.5, "Chưa đủ 5 phiên, mặc định Tài."
    if pattern[-4:] in ["TXTT", "XTXX"]:
        return ('X' if pattern[-4:] == "TXTT" else 'T'), 0.75, "Cầu 1-2-1."
    return None, 0.0, "Không tín hiệu."

def model_9_cau_2_1_2(pattern: str):
    """Phân tích cầu 2-1-2"""
    if not pattern:
        return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 6:
        return 'T', 0.5, "Chưa đủ 6 phiên, mặc định Tài."
    if pattern[-5:] in ["TTXTX", "XXTXT"]:
        return ('T' if pattern[-5:] == "XXTXT" else 'X'), 0.75, "Cầu 2-1-2."
    return None, 0.0, "Không tín hiệu."

def model_10_can_kiet(pattern: str):
    """Phân tích cân bằng dài hạn"""
    if not pattern:
        return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 20:
        return 'T', 0.5, "Chưa đủ 20 phiên, mặc định Tài."
    t_ratio = pattern[-20:].count('T') / 20
    if t_ratio >= 0.7:
        return 'X', (t_ratio - 0.6) * 2.5, "Tài cao, cân bằng Xỉu."
    if t_ratio <= 0.3:
        return 'T', (0.4 - t_ratio) * 2.5, "Xỉu cao, cân bằng Tài."
    return None, 0.0, "Cân bằng."

# --- API dự đoán VIP ---
@app.route('/predict_vip', methods=['GET'])
def predict_vip():
    global historical_data, prediction_history
    try:
        response = requests.get(SOURCE_API_URL, timeout=10)
        response.raise_for_status()
        latest_data = response.json()
        if not historical_data or historical_data[-1]['Phien'] != latest_data['Phien']:
            historical_data.append(latest_data)
            logger.info(f"Đã thêm phiên mới: {latest_data['Phien']}")
    except requests.RequestException as e:
        logger.error(f"Lỗi API nguồn: {str(e)}")
        return jsonify({"error": f"Lỗi API nguồn: {str(e)}"}), 500

    if not historical_data:
        logger.warning("Chưa có dữ liệu lịch sử.")
        return jsonify({
            "current_session": 0,
            "dice": [0, 0, 0],
            "total": 0,
            "result": "N/A",
            "next_session": 1,
            "du_doan": "Tài",
            "confidence": "50%",
            "meta": "Chưa có dữ liệu, mặc định Tài.",
            "models": [],
            "id": "Tele@HoVanThien_Pro"
        }), 200

    recent_sessions = historical_data[-200:]  # Dùng 200 phiên để phân tích, lưu full
    pattern = "".join(['T' if s['Ket_qua'] == 'Tài' else 'X' for s in recent_sessions])

    # Danh sách model ensemble
    models_config = [
        {"name": "Bệt/1-1", "func": model_1_bet_va_1_1, "weight": 1.5, "args": (pattern,)},
        {"name": "Nhịp 1-2/2-1/2-2", "func": model_2_cau_nhip, "weight": 1.5, "args": (pattern,)},
        {"name": "Thống Kê Pattern", "func": model_3_thong_ke, "weight": 1.2, "args": (pattern,)},
        {"name": "3-1/1-3", "func": model_4_cau_phuc_tap, "weight": 1.0, "args": (pattern,)},
        {"name": "4-1/1-4", "func": model_5_cau_4_1, "weight": 1.0, "args": (pattern,)},
        {"name": "2-3/3-2", "func": model_6_cau_2_3, "weight": 1.0, "args": (pattern,)},
        {"name": "Điểm Số Trend", "func": model_7_score_trend, "weight": 1.0, "args": (recent_sessions,)},
        {"name": "1-2-1", "func": model_8_cau_1_2_1, "weight": 0.9, "args": (pattern,)},
        {"name": "2-1-2", "func": model_9_cau_2_1_2, "weight": 0.9, "args": (pattern,)},
        {"name": "Cân Bằng", "func": model_10_can_kiet, "weight": 0.8, "args": (pattern,)},
        {"name": "Probability", "func": model_probability, "weight": 1.3, "args": (pattern,)},
        {"name": "Markov", "func": model_markov, "weight": 1.4, "args": (pattern,)},
        {"name": "N-gram", "func": model_ngram, "weight": 1.4, "args": (pattern,)},
        {"name": "Heuristic", "func": model_heuristic, "weight": 1.2, "args": (pattern, recent_sessions)},
    ]

    score_tai = 0.0
    score_xiu = 0.0
    model_details = []

    # Thu thập dự đoán từ các model
    for config in models_config:
        try:
            pred, conf, reason = config["func"](*config["args"])
            score = config["weight"] * conf
            if pred == 'T':
                score_tai += score
            elif pred == 'X':
                score_xiu += score
            if pred:
                model_details.append({"model": config["name"], "pred": pred, "conf": f"{conf:.2f}", "reason": reason})
        except Exception as e:
            logger.error(f"Lỗi model {config['name']}: {str(e)}")
            continue

    # Meta-Analysis ngắn gọn
    meta = ""
    if abs(score_tai - score_xiu) < 0.5:
        meta = "Điểm gần nhau, rủi ro cao."

    # Quyết định cuối cùng
    final_pred = "Tài" if score_tai >= score_xiu else "Xỉu"
    total_score = score_tai + score_xiu
    conf_val = 50 + (abs(score_tai - score_xiu) / total_score * 50) if total_score > 0 else 50
    conf_str = f"{min(98, int(conf_val))}%" 

    # Cập nhật lịch sử dự đoán
    last_session = historical_data[-1]
    if prediction_history and prediction_history[-1]["session"] == last_session['Phien']:
        prev = prediction_history[-1]
        prev["result"] = last_session['Ket_qua']
        prev["danh_gia"] = "✅" if prev["du_doan"] == last_session['Ket_qua'] else "❌"
    
    prediction_history.append({
        "session": last_session['Phien'] + 1,
        "result": None,
        "du_doan": final_pred,
        "danh_gia": None
    })

    # JSON response gọn gàng
    response = {
        "current_session": last_session['Phien'],
        "dice": [
            last_session.get('Xuc_xac_1', 0),
            last_session.get('Xuc_xac_2', 0),
            last_session.get('Xuc_xac_3', 0)
        ],
        "total": last_session.get('Tong', 0),
        "result": last_session.get('Ket_qua', 'N/A'),
        "next_session": last_session['Phien'] + 1,
        "du_doan": final_pred,
        "confidence": conf_str,
        "meta": meta,
        "models": model_details,
        "id": "Tele@HoVanThien_Pro"
    }
    logger.info(f"Dự đoán phiên {response['next_session']}: {final_pred} ({conf_str})")
    return jsonify(response)

# --- API lịch sử dự đoán ---
@app.route('/history-predict', methods=['GET'])
def history_predict():
    return jsonify({
        "history": prediction_history,
        "total": len(prediction_history),
        "id": "Tele@HoVanThien_Pro"
    })

# --- API kiểm tra trạng thái ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "id": "Tele@HoVanThien_Pro"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
