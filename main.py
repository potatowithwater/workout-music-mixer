# -*- coding: utf-8 -*-
"""
运动节拍音乐混音器（高音质版）
把本地文件夹里的 MP3 合成一首节奏统一、节拍可衔接、响度一致的连续运动音乐。

用法：python workout_music_mixer.py

必装：pip install numpy librosa
推荐：brew install rubberband && pip install pyrubberband   （高质量变速不变调）
      pip install imageio-ffmpeg                            （自带真 ffmpeg，用于导出 mp3）
      pip install pyloudnorm                                （标准 LUFS 响度测量）
"""
import os
import sys
import wave
import shutil
import subprocess
import tempfile

import numpy as np
import librosa

SR = 44100

# rubberband 高质量参数：
#   -c 5   = crisp 5，保护鼓点与瞬态（电子/流行/摇滚的运动音乐适用，0 适合古典）
#   -F     = 保留共振峰，变速后人声不发闷
RB_ARGS = ["-c", "5", "-F"]


def _probe_pyrb():
    """pip 装的 pyrubberband 只是 Python 封装，真正干活的是 rubberband 命令行程序。
    所以不能只看 import 成功，还要确认 rubberband 二进制存在且能真跑一次。"""
    try:
        import pyrubberband as pyrb
    except Exception:
        return None
    if shutil.which("rubberband") is None:
        return None
    try:
        pyrb.time_stretch(np.zeros(SR, dtype=np.float32), SR, 1.0)
        return pyrb
    except Exception:
        return None


PYRB = _probe_pyrb()
HAS_PYRB = PYRB is not None


def analyze(path):
    """读取音频（保留立体声），检测 BPM 与每个节拍的时间点（秒）。"""
    try:
        y_orig, sr = librosa.load(path, sr=SR, mono=False)
        y_mono = librosa.to_mono(y_orig)
        tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr)
        tempo = float(np.asarray(tempo).reshape(-1)[0])
        if tempo <= 0 or np.isnan(tempo):
            return None
        while tempo < 80.0:
            tempo *= 2.0
        while tempo > 190.0:
            tempo /= 2.0
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        if len(beat_times) < 8:
            return None

        # 统一内部形状为 (n,) 单声道 或 (n, 2) 立体声
        if np.asarray(y_orig).ndim == 1:
            y_data = np.ascontiguousarray(y_orig, dtype=np.float32)
        else:
            y_data = np.ascontiguousarray(np.asarray(y_orig).T, dtype=np.float32)

        return {
            "name": os.path.basename(path),
            "path": path,
            "y": y_data,
            "sr": sr,
            "bpm": tempo,
            "beats": beat_times,
            "dur": len(y_data) / float(sr),
            "stereo": y_data.ndim == 2,
        }
    except Exception as e:
        print("  [跳过] %s -> %s" % (os.path.basename(path), e))
        return None


def _rb_stretch(y, sr, rate):
    """带高质量参数调用 rubberband，参数不兼容时自动降级为默认参数。"""
    last = None
    for args in (RB_ARGS, None):
        try:
            if args is None:
                return PYRB.time_stretch(y, sr, rate)
            return PYRB.time_stretch(y, sr, rate, rbargs=args)
        except Exception as e:
            last = e
            continue
    raise last


def _ffmpeg_filter_works(ff, filt):
    """探测某个 ffmpeg 滤镜是否真的可用（有些静态构建没编进去）。"""
    try:
        import soundfile as sf
        tmp = tempfile.mkdtemp(prefix="probe_")
        y = (0.25 * np.sin(2 * np.pi * 440 * np.arange(SR) / float(SR))).astype(np.float32)
        inp = os.path.join(tmp, "i.wav")
        outp = os.path.join(tmp, "o.wav")
        sf.write(inp, y, SR, subtype="FLOAT")
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
               "-i", inp, "-filter:a", filt, "-c:a", "pcm_f32le", outp]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        ok = r.returncode == 0 and os.path.isfile(outp) and os.path.getsize(outp) > 1024
        shutil.rmtree(tmp, ignore_errors=True)
        return ok
    except Exception:
        return False


def _probe_ffmpeg_engine():
    """探测 ffmpeg 里可用的变速滤镜，优先 rubberband，其次 atempo。"""
    ff, _ = find_ffmpeg()
    if not ff:
        return None, None, None
    if _ffmpeg_filter_works(ff, "rubberband=tempo=1.1"):
        return ff, "rubberband", None
    if _ffmpeg_filter_works(ff, "atempo=1.1"):
        return ff, "atempo", None
    return None, None, "ffmpeg 里既没有 rubberband 也没有 atempo 滤镜"


# 延迟探测：find_ffmpeg 定义在后面，首次用到时才真正跑
FFMPEG_FF = None
FF_ENGINE = None
_FF_PROBED = False


def _ensure_ffmpeg_engine():
    global FFMPEG_FF, FF_ENGINE, _FF_PROBED
    if not _FF_PROBED:
        _FF_PROBED = True
        FFMPEG_FF, FF_ENGINE, _ = _probe_ffmpeg_engine()
    return FF_ENGINE


def _chain_atempo(rate):
    """atempo 单次有取值范围，超了就拆成两级串起来。"""
    r = float(rate)
    if 0.5 <= r <= 2.0:
        return "atempo=%.6f" % r
    import math
    half = math.sqrt(r)
    a = min(max(half, 0.5), 2.0)
    b = r / a
    return "atempo=%.6f,atempo=%.6f" % (a, b)


def ffmpeg_stretch(y, sr, rate, ff, engine):
    """用 ffmpeg 变速：保立体声，时间域算法，不会有相位声码器的金属味。"""
    import soundfile as sf
    data = np.asarray(y, dtype=np.float32)
    was_mono = data.ndim == 1
    tmp = tempfile.mkdtemp(prefix="ffst_")
    try:
        inp = os.path.join(tmp, "in.wav")
        outp = os.path.join(tmp, "out.wav")
        sf.write(inp, data, sr, subtype="FLOAT")
        if engine == "rubberband":
            filt = "rubberband=tempo=%.6f" % float(rate)
        else:
            filt = _chain_atempo(rate)
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
               "-i", inp, "-filter:a", filt, "-ar", str(sr), "-c:a", "pcm_f32le", outp]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600)
        if r.returncode != 0 or not os.path.isfile(outp):
            raise RuntimeError((r.stderr or b"").decode("utf-8", "ignore")[-200:])
        out, _ = sf.read(outp, dtype="float32", always_2d=False)
        out = np.asarray(out, dtype=np.float32)
        if was_mono and out.ndim == 2:
            out = np.mean(out, axis=1)
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def stretch(y, sr, rate):
    """变速不变调，rate > 1 表示加快（时长变短）。按音质优先级自动回退。"""
    global HAS_PYRB, FF_ENGINE, FFMPEG_FF
    if HAS_PYRB:
        try:
            return np.asarray(_rb_stretch(y, sr, rate), dtype=np.float32)
        except Exception:
            HAS_PYRB = False
            print("  [提示] rubberband 命令行调用失败，改用 ffmpeg 变速")
    if _ensure_ffmpeg_engine():
        try:
            return ffmpeg_stretch(y, sr, rate, FFMPEG_FF, FF_ENGINE)
        except Exception as e:
            print("  [提示] ffmpeg %s 变速失败，改用 librosa 内置引擎: %s" % (FF_ENGINE, e))
            FF_ENGINE = None
            FFMPEG_FF = None
    # 兜底：librosa 相位声码器只支持单声道，会损失立体声
    if np.asarray(y).ndim == 2:
        y = np.mean(np.asarray(y), axis=1)
    return np.asarray(librosa.effects.time_stretch(y=np.asarray(y, np.float32), rate=rate), np.float32)


def measure_lufs(y, sr):
    """按 ITU-R BS.1770 测整体响度（LUFS）。测不了返回 None。"""
    try:
        import pyloudnorm
        data = np.asarray(y, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        meter = pyloudnorm.Meter(sr)
        val = float(meter.integrated_loudness(data))
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    except Exception:
        return None


def normalize_loudness(y, sr, target_lufs):
    """把音频整体增益到目标响度。gain 设上限，避免静音段被拉爆。"""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    lu = measure_lufs(y, sr)
    if lu is None:
        # 无 pyloudnorm 时用 RMS 近似（对音乐而言误差约 1~2 dB，够用）
        rms = float(np.sqrt(np.mean(np.square(y))))
        if rms <= 1e-9:
            return y
        target_rms = 10.0 ** ((target_lufs - 14.0) / 20.0)
        gain = target_rms / rms
    else:
        gain = 10.0 ** ((target_lufs - lu) / 20.0)
    gain = float(min(gain, 10.0 ** (36.0 / 20.0)))
    return (y * np.float32(gain)).astype(np.float32)


def soft_limit(y, ceiling_dbfs=-1.0):
    """软削波限幅：小信号不变，超过阈值的部分平滑压回，峰值严格不超过 ceiling。"""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    ceiling = 10.0 ** (ceiling_dbfs / 20.0)
    peak = float(np.max(np.abs(y)))
    if peak <= ceiling or peak == 0.0:
        return y
    x = y / peak
    a = np.abs(x)
    th = 0.75
    shaped = np.where(a <= th, a, th + (1.0 - th) * np.tanh((a - th) / (1.0 - th)))
    top = th + (1.0 - th) * np.tanh(1.0)
    return (np.sign(x) * shaped / top * ceiling).astype(np.float32)


def normalize_and_limit(y, sr, target_lufs, ceiling_dbfs=-1.0, iters=5):
    """归一化 + 限幅 + 补增益，反复逼近目标响度。

    只做一次的话，限幅压掉峰值会连带把整体响度拉低好几 dB，
    所以这里补增益再限幅、迭代收敛，保证最终既够响又不削波。
    """
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    y = normalize_loudness(y, sr, target_lufs)
    final = y
    for _ in range(iters):
        lim = soft_limit(y, ceiling_dbfs)
        lu = measure_lufs(lim, sr)
        final = lim
        if lu is None:
            break
        diff = target_lufs - lu
        if abs(diff) < 0.3:
            break
        # 补增益（限幅会压低响度），下一轮由 soft_limit 把峰值压回 ceiling
        gain = 10.0 ** (min(diff, 6.0) / 20.0)
        y = (lim * np.float32(gain)).astype(np.float32)
        final = y
    return soft_limit(final, ceiling_dbfs)


def build_segment(y, beats, bpm, target_bpm, tail_beats=2):
    """把一首歌伸缩到目标 BPM，并裁成以第一拍开头、结尾留两拍尾巴的片段。"""
    rate = target_bpm / bpm
    if rate < 0.8 or rate > 1.25:
        return None
    y2 = stretch(y, SR, rate)
    beats2 = beats / rate
    bi = 60.0 / target_bpm

    start_sec = float(beats2[0])
    end_sec = min(len(y2) / float(SR), float(beats2[-1]) + tail_beats * bi)
    s0 = int(max(0.0, start_sec) * SR)
    s1 = int(end_sec * SR)
    if s1 - s0 < SR:
        return None
    y_cut = np.asarray(y2, dtype=np.float32)[s0:s1]
    beats_rel = beats2 - start_sec
    beats_rel = beats_rel[(beats_rel >= 0) & (beats_rel < len(y_cut) / float(SR))]
    if len(beats_rel) < 4:
        return None
    return y_cut, beats_rel


def layout(segments, target_bpm):
    """计算每首歌的起始时间：让后一首第一拍落在前一首最后一拍之后正好一拍。"""
    bi = 60.0 / target_bpm
    starts = [0.0]
    for i in range(1, len(segments)):
        prev_start = starts[i - 1]
        prev_beats = segments[i - 1][1]
        starts.append(prev_start + float(prev_beats[-1]) + bi)
    return starts


def mix(segments, starts, target_bpm, sr=SR):
    """铺进总轨，重叠的一拍内做交叉淡化。"""
    bi = 60.0 / target_bpm
    fade = int(bi * sr)

    nch = 1
    for y_cut, _ in segments:
        if np.asarray(y_cut).ndim == 2:
            nch = max(nch, np.asarray(y_cut).shape[1])

    total = 0
    for i, (y_cut, _) in enumerate(segments):
        total = max(total, int(starts[i] * sr) + len(np.asarray(y_cut)))

    if nch == 1:
        out = np.zeros(total, dtype=np.float32)
    else:
        out = np.zeros((total, nch), dtype=np.float32)

    for i, (y_cut, _) in enumerate(segments):
        pos = int(starts[i] * sr)
        n = len(np.asarray(y_cut))
        buf = np.asarray(y_cut, dtype=np.float32).copy()
        if buf.ndim == 1 and nch == 2:
            buf = np.stack([buf, buf], axis=1)
        if i > 0:
            f = min(fade, n // 2)
            if f > 0:
                ramp = np.linspace(0.0, 1.0, f).astype(np.float32)
                buf[:f] = buf[:f] * (ramp[:, None] if buf.ndim == 2 else ramp)
        if i < len(segments) - 1:
            f = min(fade, n // 2)
            if f > 0:
                ramp = np.linspace(1.0, 0.0, f).astype(np.float32)
                buf[-f:] = buf[-f:] * (ramp[:, None] if buf.ndim == 2 else ramp)
        out[pos:pos + n] += buf

    return out


def write_wav(path, data, sr):
    """用标准库写 16bit WAV，自动识别单声道/立体声。"""
    d = np.asarray(data, dtype=np.float32)
    ch = 1 if d.ndim == 1 else d.shape[1]
    pcm = (np.clip(d, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _ffmpeg_works(exe):
    """验证这真的是 ffmpeg 二进制，而不是 pip 里那个同名的空壳包装器。"""
    try:
        r = subprocess.run([exe, "-version"], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=15)
        text = (r.stdout or b"").decode("utf-8", "ignore") + (r.stderr or b"").decode("utf-8", "ignore")
        return r.returncode == 0 and "ffmpeg version" in text
    except Exception:
        return False


def find_ffmpeg():
    """按优先级找可用的 ffmpeg 二进制，返回 (路径 or None, 诊断记录)。"""
    notes = []
    candidates = []

    for name in ("ffmpeg", "ffmpeg.exe"):
        p = shutil.which(name)
        if p:
            candidates.append(("PATH", p))

    try:
        base = os.path.dirname(os.path.dirname(sys.executable))
        for sub in ("bin", "Scripts", os.path.join("Library", "bin")):
            for name in ("ffmpeg", "ffmpeg.exe"):
                candidates.append(("venv/%s" % sub, os.path.join(base, sub, name)))
    except Exception:
        pass

    for p in ("/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg", "/usr/bin/ffmpeg",
              "/opt/local/bin/ffmpeg", r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"):
        candidates.append(("常见路径", p))

    for mod in ("imageio_ffmpeg", "static_ffmpeg"):
        try:
            m = __import__(mod)
            exe = m.get_ffmpeg_exe() if hasattr(m, "get_ffmpeg_exe") else m.get_or_fetch_platform_executables_else_raise()[0]
            candidates.append((mod, exe))
        except Exception:
            notes.append("未安装 %s" % mod)

    seen = set()
    for source, p in candidates:
        if not p or p in seen:
            continue
        seen.add(p)
        if not (os.path.isfile(p) and os.access(p, os.X_OK)):
            continue
        if _ffmpeg_works(p):
            notes.append("命中 %s: %s" % (source, p))
            return p, notes
        notes.append("%s 不是真 ffmpeg: %s" % (source, p))

    return None, notes


def export_mp3(wav_path, mp3_path, bitrate="320k"):
    """WAV 转 MP3：优先 ffmpeg，其次 pydub。返回 (是否成功, 失败原因)。"""
    ff, notes = find_ffmpeg()
    last_err = ""
    if ff:
        for codec in ("libmp3lame", "mp3"):
            cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
                   "-i", wav_path, "-codec:a", codec, "-b:a", bitrate,
                   "-ar", "44100", "-map_metadata", "-1", mp3_path]
            try:
                r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600)
                if os.path.isfile(mp3_path) and os.path.getsize(mp3_path) > 1024:
                    return True, ""
                last_err = (r.stderr or b"").decode("utf-8", "ignore").strip()[-300:]
            except Exception as e:
                last_err = str(e)
        return False, ("ffmpeg 转码失败：%s" % (last_err or "未知原因"))

    try:
        from pydub import AudioSegment
        AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3", bitrate=bitrate)
        if os.path.isfile(mp3_path) and os.path.getsize(mp3_path) > 1024:
            return True, ""
    except Exception as e:
        last_err = str(e)

    detail = "；".join(notes[-4:]) if notes else "无"
    return False, ("找不到可用的 ffmpeg。诊断：%s%s" % (detail, ("；pydub: %s" % last_err) if last_err else ""))


def pick_songs(songs, target_bpm, auto=True, tol=0.15):
    if auto:
        lo, hi = target_bpm * (1 - tol), target_bpm * (1 + tol)
        sel = [s for s in songs if lo <= s["bpm"] <= hi]
        return sel if sel else songs
    print("可多选，用逗号分隔，例如 0,2,5")
    raw = input("请输入序号: ").strip()
    idx = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            j = int(part)
            if 0 <= j < len(songs):
                idx.append(j)
    return [songs[j] for j in idx]


def ask_loudness():
    """让用户选目标响度。数值越大越响，但动态被压得越多。"""
    print("")
    print("响度目标（LUFS，数值越大越响）：")
    print("  [1] -16  温和，保留最多动态，安静环境听")
    print("  [2] -13  适中，接近流媒体标准（推荐）")
    print("  [3] -11  响，健身房/椭圆机合适")
    print("  [4] -9   很响，户外跑步或环境嘈杂时用")
    raw = input("请选择 [3]: ").strip() or "3"
    table = {"1": -16.0, "2": -13.0, "3": -11.0, "4": -9.0}
    return table.get(raw, -11.0)


def main():
    print("=== 运动节拍音乐混音器（高音质版）===")
    _ensure_ffmpeg_engine()
    if HAS_PYRB:
        eng = "rubberband 命令行（最佳，立体声）"
    elif FF_ENGINE == "rubberband":
        eng = "ffmpeg rubberband 滤镜（佳，立体声）"
    elif FF_ENGINE == "atempo":
        eng = "ffmpeg atempo（良好，立体声，无金属味）"
    else:
        eng = "librosa 相位声码器（一般，单声道）"
    print("变速引擎: %s" % eng)
    if not HAS_PYRB:
        print("  想要最佳音质：安装 Homebrew 后 brew install rubberband && pip install pyrubberband")

    _ff, _notes = find_ffmpeg()
    if _ff:
        print("转码器: %s" % _ff)
    else:
        print("转码器: 未找到真正的 ffmpeg（将输出 WAV）")
        print("  修复：pip install imageio-ffmpeg 然后重跑本脚本")

    folder = input("请输入 MP3 文件夹路径: ").strip().strip('"')
    if not os.path.isdir(folder):
        print("路径无效")
        sys.exit(1)

    files = [f for f in os.listdir(folder) if f.lower().endswith(".mp3")]
    if not files:
        print("文件夹内没有 mp3")
        sys.exit(1)

    print("")
    print("共发现 %d 个文件，开始分析 BPM..." % len(files))
    songs = []
    for f in files:
        info = analyze(os.path.join(folder, f))
        if info:
            songs.append(info)
            print("  %-40s %6.1f BPM  %5.0f 秒 %s" % (
                info["name"][:40], info["bpm"], info["dur"], "立体声" if info["stereo"] else "单声道"))

    if not songs:
        print("没有可分析的歌曲")
        sys.exit(1)

    songs.sort(key=lambda s: s["bpm"])
    print("")
    print("分析结果：")
    for i, s in enumerate(songs):
        print("  [%d] %s  %.1f BPM  (%.0f 秒)" % (i, s["name"], s["bpm"], s["dur"]))

    try:
        base = int(input("\n选择与你步频最一致的基准歌曲序号: ").strip())
        target = songs[base]["bpm"]
    except Exception:
        print("输入无效")
        sys.exit(1)
    print("目标 BPM = %.1f" % target)

    mode = input("匹配模式：1=自动（±15%） 2=手动  [1] ").strip() or "1"
    selected = pick_songs(songs, target, auto=(mode == "1"))
    if not selected:
        print("未选中歌曲")
        sys.exit(1)
    print("选中 %d 首" % len(selected))

    target_lufs = ask_loudness()
    print("目标响度: %.0f LUFS，真峰值上限 -1.0 dBTP" % target_lufs)

    print("")
    print("开始变速、逐首响度对齐与节拍对齐...")
    segments = []
    for s in selected:
        seg = build_segment(s["y"], s["beats"], s["bpm"], target)
        if seg is None:
            print("  跳过 %s（变速超出安全范围或节拍不足）" % s["name"])
            continue
        y_cut, beats_rel = seg
        # 逐首先对齐到同一响度，避免衔接处音量忽大忽小
        y_cut = normalize_loudness(y_cut, SR, target_lufs)
        segments.append((y_cut, beats_rel))
        print("  就绪 %s  %.1f -> %.1f BPM" % (s["name"], s["bpm"], target))

    if not segments:
        print("没有可用片段")
        sys.exit(1)

    starts = layout(segments, target)
    print("正在合成...")
    out = mix(segments, starts, target)

    # 整体再归一化并限幅，保证成品响度达标且不削波失真
    out = normalize_and_limit(out, SR, target_lufs, ceiling_dbfs=-1.0)

    total_sec = len(out) / float(SR)
    print("合成完成，总时长 %.1f 秒（%.1f 分钟）" % (total_sec, total_sec / 60.0))

    lu = measure_lufs(out, SR)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    peak_db = 20.0 * np.log10(peak) if peak > 0 else -99.0
    print("成品响度: %s，真峰值: %.1f dBTP" % (
        ("%.1f LUFS" % lu) if lu is not None else "未测（未装 pyloudnorm）", peak_db))
    if lu is None:
        print("  装 pyloudnorm 可显示精确响度：pip install pyloudnorm")

    bitrate = input("MP3 码率 [320k]（可选 192k/256k/320k）: ").strip() or "320k"
    if not bitrate.endswith("k"):
        bitrate = "320k"

    tmp_dir = tempfile.mkdtemp(prefix="workout_mix_")
    out_wav = os.path.join(tmp_dir, "mix.wav")
    write_wav(out_wav, out, SR)

    out_mp3 = os.path.join(folder, "workout_mix_%dBPM.mp3" % int(target))
    n = 1
    while os.path.exists(out_mp3):
        out_mp3 = os.path.join(folder, "workout_mix_%dBPM_%d.mp3" % (int(target), n))
        n += 1

    print("正在转 MP3（%s）..." % bitrate)
    ok, reason = export_mp3(out_wav, out_mp3, bitrate)
    if ok:
        print("已输出 MP3: %s" % out_mp3)
        print("文件大小: %.1f MB" % (os.path.getsize(out_mp3) / 1024.0 / 1024.0))
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        out_wav_keep = os.path.join(folder, "workout_mix_%dBPM.wav" % int(target))
        shutil.move(out_wav, out_wav_keep)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("未能转 MP3，已保留 WAV: %s" % out_wav_keep)
        print("原因：%s" % reason)
        print("")
        print("注意：PyPI 上那个叫 ffmpeg 的包（pip install ffmpeg）不是真正的转码器，")
        print("它只是个包装器，装了也没用。请用下面任一方式装真正的 ffmpeg：")
        print("  最省事（推荐）: pip install imageio-ffmpeg      # 自带真二进制，装完直接重跑本脚本")
        print("  macOS:          brew install ffmpeg")
        print("  Windows:        去 gyan.dev 下载 essentials 版，把 bin 目录加进 PATH")
        print("  Linux:          sudo apt install ffmpeg")

    print("完成")


if __name__ == "__main__":
    main()
