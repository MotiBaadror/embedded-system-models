from dataclasses import dataclass

from _pytest.fixtures import fixture

from dir_configs import add_rootpath
from models.transformers_model.inference_exported_model import OnnxRunner


class TestOnnxRunner():
    @fixture
    def result(self):
        @dataclass
        class Result:
            runner = OnnxRunner(add_rootpath('data/model_repository/my_model_bs_1.onnx'))

        return Result()


    def test_postprocess(self,result):
        data = [[-1.0663654 , 1.7281588]]
        out = result.runner.postprocess(data)
        print(out)
        assert  isinstance(out, dict)
        assert out['isScam'] == 'false'
