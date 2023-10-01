import json
import os
from datetime import datetime
from typing import Callable, Optional

from llm_app import BaseApp, InputFormat, OutputFormat
from pydantic import BaseModel
from tqdm import tqdm


class TestCase(BaseModel):
    test_id: str
    test_input: InputFormat
    reference_output: Optional[OutputFormat] = None
    historical_output: Optional[OutputFormat] = None
    historical_feedback: Optional[str] = None


class EvalProperty(BaseModel):
    property_name: str
    description: str
    eval_func: Callable


class PropertyResult(BaseModel):
    feedback: str
    pass_fail: bool
    property_name: Optional[str] = None


class TestCaseResult(BaseModel):
    test_case_id: str
    output: OutputFormat
    property_results: list[PropertyResult]


class Evaluator:
    def __init__(
        self,
        test_set: list[TestCase],
        properties: list[EvalProperty],
        results_dir: str = "eval_results",
    ):
        self.test_set = test_set
        self.properties = properties
        self.results_dir = results_dir

    def evaluate(
        self,
        llm_app: BaseApp,
        exp_name: Optional[str] = None,
        exp_descr: str = "",
    ):
        # If no experiment name is provided, use the current timestamp
        if not exp_name:
            exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create experiment directory
        exp_dir = os.path.join(self.results_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        # Loop over test cases
        for test_case in tqdm(
            self.test_set, desc="Evaluating test cases", unit="test case", total=len(self.test_set)
        ):
            # Pass the test case to the LLM app
            app_output = llm_app(app_input=test_case.test_input)
            # Evaluate properties
            property_results = []
            for prop in self.properties:
                print(f"Evaluating property {prop.property_name}")
                r = prop.eval_func(test_case=test_case, llm_app_result=app_output)
                # If the property is None, then it is not applicable to this test case, so skip it
                if r:
                    # Store the property results per test case in a list
                    property_results.append(
                        PropertyResult(
                            property_name=prop.property_name,
                            feedback=r.feedback,
                            pass_fail=r.pass_fail,
                        )
                    )
            # Store results as JSON
            tcr = TestCaseResult(
                test_case_id=test_case.test_id, output=app_output, property_results=property_results
            )
            tcr_json = tcr.model_dump_json()
            with open(os.path.join(exp_dir, f"{tcr.test_case_id}.json"), "w") as f:
                f.write(tcr_json)
        # Save the Llm app config dict as JSON
        with open(os.path.join(exp_dir, "llm_app.json"), "w") as f:
            f.write(json.dumps(llm_app.cfg))
