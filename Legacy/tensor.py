import tensorflow as tf
import numpy as np
from pathlib import Path
import json


class TensorFlowModelAnalyzer:
    def __init__(self):
        """TensorFlow .pb 모델 파일 분석기"""
        self.analyzed_models = {}

    def load_pb_model(self, model_path):
        """
        .pb 파일을 로드하고 GraphDef를 반환
        """
        try:
            with tf.io.gfile.GFile(str(model_path), "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())

            # 그래프 생성
            graph = tf.Graph()
            with graph.as_default():
                tf.import_graph_def(graph_def, name="")

            return graph, graph_def

        except Exception as e:
            print(f"❌ 모델 로드 실패 ({model_path}): {e}")
            return None, None

    def analyze_model_structure(self, model_path, model_name):
        """
        모델의 전체 구조를 분석
        """
        print(f"\n{'=' * 60}")
        print(f"🔍 {model_name} 분석 시작")
        print(f"📁 파일: {model_path}")
        print(f"{'=' * 60}")

        graph, graph_def = self.load_pb_model(model_path)
        if graph is None:
            return None

        analysis_result = {
            "model_name": model_name,
            "model_path": str(model_path),
            "total_operations": len(graph_def.node),
            "operations": [],
            "placeholders": [],
            "outputs": [],
            "input_tensors": [],
            "output_tensors": []
        }

        print(f"📊 총 연산 수: {len(graph_def.node)}")
        print(f"\n📋 모든 연산 목록:")

        # 모든 연산 분석
        for i, node in enumerate(graph_def.node, 1):
            op_info = {
                "index": i,
                "name": node.name,
                "type": node.op,
                "inputs": list(node.input),
                "outputs": []
            }

            # 출력 텐서 정보 가져오기
            try:
                with graph.as_default():
                    # 그래프에서 해당 연산의 출력들 찾기
                    operation = graph.get_operation_by_name(node.name)
                    for j, output in enumerate(operation.outputs):
                        output_info = {
                            "name": output.name,
                            "shape": output.shape.as_list() if output.shape.is_fully_defined() else [
                                None if dim.value is None else dim.value for dim in output.shape.dims],
                            "dtype": str(output.dtype)
                        }
                        op_info["outputs"].append(output_info)
            except:
                # 연산을 찾을 수 없는 경우 (일부 내부 연산들)
                pass

            analysis_result["operations"].append(op_info)

            # Placeholder 타입 찾기
            if node.op == "Placeholder":
                placeholder_info = {
                    "name": node.name,
                    "outputs": op_info["outputs"]
                }
                analysis_result["placeholders"].append(placeholder_info)
                analysis_result["input_tensors"].extend([out["name"] for out in op_info["outputs"]])

            # 출력 형태의 연산들 찾기 (일반적으로 마지막 연산들)
            output_types = ["Sigmoid", "Softmax", "Identity", "PartitionedCall", "StatefulPartitionedCall"]
            if node.op in output_types:
                output_info = {
                    "name": node.name,
                    "type": node.op,
                    "outputs": op_info["outputs"]
                }
                analysis_result["outputs"].append(output_info)
                analysis_result["output_tensors"].extend([out["name"] for out in op_info["outputs"]])

            # 연산 정보 출력
            print(f"   {i:3d}. {node.name} (타입: {node.op})")
            for output in op_info["outputs"]:
                print(
                    f"       출력 {op_info['outputs'].index(output)}: {output['name']} - 모양: {output['shape']}, 타입: {output['dtype']}")

        # Placeholder 정보 출력
        print(f"\n🎯 입력 노드들 (Placeholder):")
        if analysis_result["placeholders"]:
            for i, placeholder in enumerate(analysis_result["placeholders"], 1):
                print(f"   {i}. {placeholder['name']}")
                for output in placeholder["outputs"]:
                    print(f"      • {output['name']} - 모양: {output['shape']}, 타입: {output['dtype']}")
        else:
            print("   입력 노드를 찾을 수 없습니다.")

        # 출력 노드 정보
        print(f"\n🎯 출력 노드들:")
        if analysis_result["outputs"]:
            for i, output in enumerate(analysis_result["outputs"], 1):
                print(f"   {i}. {output['name']} (타입: {output['type']})")
                for out_tensor in output["outputs"]:
                    print(f"      • {out_tensor['name']} - 모양: {out_tensor['shape']}, 타입: {out_tensor['dtype']}")
        else:
            print("   출력 노드를 찾을 수 없습니다.")

        # 추천 입출력 텐서
        print(f"\n💡 추천 입출력 텐서:")
        if analysis_result["input_tensors"]:
            print(f"   📥 입력 텐서: {analysis_result['input_tensors'][0]}")
        if analysis_result["output_tensors"]:
            # 가장 가능성 높은 출력 텐서 선택
            recommended_output = None
            for tensor_name in analysis_result["output_tensors"]:
                if "Sigmoid" in tensor_name or "Softmax" in tensor_name:
                    recommended_output = tensor_name
                    break
            if not recommended_output and analysis_result["output_tensors"]:
                recommended_output = analysis_result["output_tensors"][-1]

            print(f"   📤 출력 텐서: {recommended_output}")

        # 결과 저장
        self.analyzed_models[model_name] = analysis_result
        return analysis_result

    def find_serving_signatures(self, model_path, model_name):
        """
        SavedModel 형태의 서빙 시그니처를 찾기 (일부 .pb 파일에서 가능)
        """
        try:
            graph, _ = self.load_pb_model(model_path)
            if graph is None:
                return

            print(f"\n🔍 {model_name} 서빙 시그니처 분석:")

            # serving_default로 시작하는 노드들 찾기
            serving_nodes = []
            with graph.as_default():
                for op in graph.get_operations():
                    if "serving_default" in op.name:
                        serving_nodes.append(op)

            if serving_nodes:
                print("   📡 서빙 관련 노드들:")
                for node in serving_nodes:
                    print(f"      • {node.name} (타입: {node.type})")
                    for output in node.outputs:
                        print(f"        출력: {output.name} - 모양: {output.shape}")
            else:
                print("   서빙 시그니처를 찾을 수 없습니다.")

        except Exception as e:
            print(f"   서빙 시그니처 분석 실패: {e}")

    def save_analysis_to_file(self, model_name, output_file=None):
        """
        분석 결과를 파일로 저장
        """
        if model_name not in self.analyzed_models:
            print(f"❌ {model_name} 모델 분석 결과가 없습니다.")
            return

        if output_file is None:
            output_file = f"{model_name}_analysis.txt"

        result = self.analyzed_models[model_name]

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {model_name} 모델 분석 결과 ===\n\n")
            f.write(f"총 연산 수: {result['total_operations']}\n\n")

            f.write("=== 모든 연산 목록 ===\n")
            for op in result['operations']:
                f.write(f"   {op['index']}. {op['name']} (타입: {op['type']})\n")
                for output in op['outputs']:
                    f.write(
                        f"       출력 {op['outputs'].index(output)}: {output['name']} - 모양: {output['shape']}, 타입: {output['dtype']}\n")
                f.write("\n")

        print(f"✅ 분석 결과가 {output_file}에 저장되었습니다.")

    def compare_models(self):
        """
        분석된 모델들을 비교
        """
        if len(self.analyzed_models) < 2:
            print("비교할 모델이 충분하지 않습니다.")
            return

        print(f"\n{'=' * 60}")
        print("🔄 모델 비교 분석")
        print(f"{'=' * 60}")

        for name, result in self.analyzed_models.items():
            print(f"\n📋 {name}:")
            print(f"   • 총 연산 수: {result['total_operations']}")
            print(f"   • 입력 텐서 수: {len(result['input_tensors'])}")
            print(f"   • 출력 텐서 수: {len(result['output_tensors'])}")

            if result['input_tensors']:
                print(f"   • 주요 입력: {result['input_tensors'][0]}")
            if result['output_tensors']:
                print(f"   • 주요 출력: {result['output_tensors'][-1]}")


def main():
    """
    메인 실행 함수
    """
    analyzer = TensorFlowModelAnalyzer()

    # 분석할 모델들 정의
    models_to_analyze = [
        {
            "path": "models/discogs-effnet-bs64-1.pb",
            "name": "discogs-effnet-bs64-1"
        },
        {
            "path": "models/mtg_jamendo_moodtheme-discogs-effnet-1.pb",
            "name": "mtg_jamendo_moodtheme-discogs-effnet-1"
        },
        {
            "path": "models/genre_discogs400-discogs-effnet-1.pb",
            "name": "genre_discogs400-discogs-effnet-1"
        }
    ]

    print("🔍 TensorFlow .pb 모델 구조 분석기")
    print("특히 장르 모델의 올바른 입력 텐서 이름 찾기에 중점")
    print("=" * 70)

    # 각 모델 분석
    for model_info in models_to_analyze:
        model_path = Path(model_info["path"])

        if model_path.exists():
            # 기본 구조 분석
            result = analyzer.analyze_model_structure(model_path, model_info["name"])

            # 서빙 시그니처 분석
            analyzer.find_serving_signatures(model_path, model_info["name"])

            # 결과를 파일로 저장
            analyzer.save_analysis_to_file(model_info["name"])

        else:
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")

    # 모델들 비교
    analyzer.compare_models()

    print(f"\n{'=' * 70}")
    print("✅ 모든 모델 분석 완료!")
    print("각 모델의 상세 분석 결과는 개별 텍스트 파일로 저장되었습니다.")


if __name__ == "__main__":
    main()