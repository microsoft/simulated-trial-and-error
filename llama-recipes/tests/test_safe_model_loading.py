import ast
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch


RECIPES = Path(__file__).resolve().parents[1]


class SafeModelLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.peft = MagicMock()
        cls.transformers = MagicMock()
        cls.hub = MagicMock()
        with patch.dict(sys.modules, {
            "peft": cls.peft,
            "transformers": cls.transformers,
            "huggingface_hub": cls.hub,
        }):
            spec = importlib.util.spec_from_file_location(
                "model_utils_under_test", RECIPES / "inference/model_utils.py"
            )
            cls.model_utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.model_utils)

    def setUp(self):
        self.peft.reset_mock()
        self.transformers.reset_mock()
        self.hub.reset_mock()
        self.model_utils.snapshot_download.reset_mock()

    def test_base_model_requires_safetensors(self):
        self.model_utils.load_model("example/model", False)
        self.transformers.LlamaForCausalLM.from_pretrained.assert_called_once_with(
            "example/model",
            return_dict=True,
            load_in_8bit=False,
            device_map="auto",
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

    def test_remote_adapter_downloads_only_safe_files(self):
        self.model_utils.snapshot_download.return_value = "/cached/adapter"
        with patch.object(Path, "is_dir", return_value=False), patch.object(
            Path, "is_file", return_value=True
        ):
            self.model_utils.load_peft_model("base", "example/adapter")
        self.model_utils.snapshot_download.assert_called_once_with(
            repo_id="example/adapter",
            allow_patterns=["*.json", "*.safetensors"],
        )
        self.peft.PeftModel.from_pretrained.assert_called_once_with(
            "base", "/cached/adapter"
        )

    def test_adapter_without_safetensors_is_rejected(self):
        with patch.object(Path, "is_dir", return_value=True), patch.object(
            Path, "is_file", return_value=False
        ):
            with self.assertRaisesRegex(ValueError, "adapter_model.safetensors"):
                self.model_utils.load_peft_model("base", "/legacy/adapter")
        self.peft.PeftModel.from_pretrained.assert_not_called()

    def test_full_checkpoints_restrict_pickle(self):
        source = (RECIPES / "model_checkpointing/checkpoint_handler.py").read_text()
        functions = [
            node for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name in {
                "load_model_checkpoint", "load_optimizer_checkpoint"
            }
        ]
        torch = MagicMock()
        fsdp = MagicMock()
        namespace = {"Path": Path, "torch": torch, "FSDP": fsdp}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<checkpoints>", "exec"), namespace)

        cfg = MagicMock(checkpoint_folder="checkpoints",
                        checkpoint_model_filename="model.pt",
                        optimizer_checkpoint_file="optimizer.pt",
                        verbose=False)
        with patch.object(Path, "is_file", return_value=True):
            namespace["load_model_checkpoint"](MagicMock(), 0, cfg)
            namespace["load_optimizer_checkpoint"](MagicMock(), MagicMock(), 0, cfg)
        self.assertEqual(torch.load.call_count, 2)
        for call in torch.load.call_args_list:
            self.assertIs(call.kwargs["weights_only"], True)


if __name__ == "__main__":
    unittest.main()
