import pytest

from psr.manifest import Manifest, stage_key


class TestStageKey:
    def test_stable_across_dict_key_ordering(self):
        a = stage_key("asr", "v1", ["hash-abc"], {"engine": "groq", "temperature": 0.0})
        b = stage_key("asr", "v1", ["hash-abc"], {"temperature": 0.0, "engine": "groq"})
        assert a == b

    def test_stable_across_nested_dict_ordering(self):
        params_a = {"decode": {"beam_size": 1, "language": "zh"}, "engine": "groq"}
        params_b = {"engine": "groq", "decode": {"language": "zh", "beam_size": 1}}
        assert stage_key("asr", "v1", ["h"], params_a) == stage_key("asr", "v1", ["h"], params_b)

    def test_changes_when_stage_name_changes(self):
        a = stage_key("asr", "v1", ["h"], {})
        b = stage_key("polish", "v1", ["h"], {})
        assert a != b

    def test_changes_when_version_changes(self):
        a = stage_key("asr", "v1", ["h"], {})
        b = stage_key("asr", "v2", ["h"], {})
        assert a != b

    def test_changes_when_inputs_change(self):
        a = stage_key("asr", "v1", ["hash-a"], {})
        b = stage_key("asr", "v1", ["hash-b"], {})
        assert a != b

    def test_changes_when_input_order_changes(self):
        a = stage_key("align", "v1", ["h1", "h2"], {})
        b = stage_key("align", "v1", ["h2", "h1"], {})
        assert a != b

    def test_changes_when_any_param_value_changes(self):
        a = stage_key("polish", "v1", ["h"], {"model": "gpt-oss-120b", "seed": 42})
        b = stage_key("polish", "v1", ["h"], {"model": "gpt-oss-120b", "seed": 43})
        assert a != b

    def test_deterministic_hex_digest_truncated_to_16(self):
        key = stage_key("audio", "v1", ["md5-xyz"], {"format": "16k-mono-mp3-32k"})
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)
        # 重跑同樣的輸入必須拿到逐位元相同的 key（決定性快取的基礎）。
        assert key == stage_key("audio", "v1", ["md5-xyz"], {"format": "16k-mono-mp3-32k"})


class TestManifest:
    def test_json_round_trip(self):
        m = Manifest(
            source="dQw4w9WgXcQ",
            source_md5="abc123",
            stage_keys={"audio": "1111111111111111", "asr": "2222222222222222"},
            engine="groq",
            model_ids={"asr": "whisper-large-v3-turbo", "polish": "openai/gpt-oss-120b"},
            model_revisions={"asr": "rev-1"},
            gpu_model="T4",
            library_versions={"faster-whisper": "1.0.0", "ffmpeg": "6.1"},
            timings={"audio": 11.2, "asr": 63.4},
            cost=0.12,
            degraded_window_count=1,
        )
        restored = Manifest.from_json(m.to_json())
        assert restored == m

    def test_round_trip_preserves_defaults(self):
        m = Manifest(source="fileid123")
        restored = Manifest.from_json(m.to_json())
        assert restored == m
        assert restored.stage_keys == {}
        assert restored.cost == 0.0

    def test_is_complete_true_when_key_matches(self):
        m = Manifest(source="x", stage_keys={"audio": "deadbeefdeadbeef"})
        assert m.is_complete("audio", "deadbeefdeadbeef") is True

    def test_is_complete_false_when_key_differs(self):
        m = Manifest(source="x", stage_keys={"audio": "deadbeefdeadbeef"})
        assert m.is_complete("audio", "0000000000000000") is False

    def test_is_complete_false_when_stage_missing(self):
        m = Manifest(source="x")
        assert m.is_complete("audio", "deadbeefdeadbeef") is False

    def test_gpu_model_and_library_versions_do_not_affect_stage_key(self):
        # 設計文件 §5：GPU 型號跟套件版本寫進 manifest 純粹是為了除錯，
        # 完全不進 key 計算——同一份輸入不管在哪張卡上跑，key 都一樣。
        key = stage_key("asr", "v1", ["h"], {"engine": "colab"})
        m1 = Manifest(source="x", gpu_model="T4")
        m2 = Manifest(source="x", gpu_model="A100")
        assert stage_key("asr", "v1", ["h"], {"engine": "colab"}) == key
        assert m1.gpu_model != m2.gpu_model  # 只是確認欄位本身照樣能記錄差異
