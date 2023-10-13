import json
import os
import time
from datetime import datetime
from typing import Callable, Optional

import pandas as pd
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
    feedback: Optional[str]
    score: float


class TestCaseResult(BaseModel):
    test_case_id: str
    output: OutputFormat
    property_results: dict[str, PropertyResult]
    latency: float
    cosine_similarity: Optional[float] = None
    verbosity: Optional[float] = None


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
        test_case_results = []
        for test_case in tqdm(
            self.test_set, desc="Evaluating test cases", unit="test case", total=len(self.test_set)
        ):
            # Pass the test case to the LLM app
            # Measure the time it takes to run the LLM app
            start_time = time.time()
            app_output = llm_app(app_input=test_case.test_input)
            latency = time.time() - start_time
            # Evaluate properties
            property_results = {}
            for prop in self.properties:
                # print(f"Evaluating property {prop.property_name}")
                r = prop.eval_func(test_case=test_case, llm_app_result=app_output)
                # If the property is None, then it is not applicable to this test case, so skip it
                if r:
                    # Store the property results per test case in a list
                    property_results[prop.property_name] = PropertyResult(
                        feedback=r.feedback,
                        score=r.score if "score" in r.model_fields else float(r.pass_fail),
                    )
            # Store results as JSON
            tcr = TestCaseResult(
                test_case_id=test_case.test_id,
                output=app_output,
                property_results=property_results,
                latency=latency,
            )
            test_case_results.append(tcr)
            tcr_json = tcr.model_dump_json()
            with open(os.path.join(exp_dir, f"{tcr.test_case_id}.json"), "w") as f:
                f.write(tcr_json)
        # Save the Llm app config dict as JSON
        with open(os.path.join(exp_dir, "llm_app.json"), "w") as f:
            f.write(json.dumps(llm_app.cfg))

        # Convert all test case results into a dataframe
        df = pd.DataFrame([tcr.dict() for tcr in test_case_results])
        # Split the `property_results` into separate columns. The `property_results` column is a dict of dicts.
        # Each top level key is a property name. Each second level key is a property result (feedback and pass_fail).
        # The `property_results` column is split into separate columns for each combination of property name and property result.
        # The values of these columns are the values of the `feedback` and `pass_fail` respectively.
        df = df.join(pd.json_normalize(df["property_results"]))
        # Drop the `property_results` column.
        df = df.drop(columns=["property_results"])
        # Save the dataframe as CSV
        df.to_csv(os.path.join(exp_dir, "results.csv"), index=False)
        # Aggregate the results by taking the mean over the test cases for the `latency` and all `pass_fail` columns.
        agg_columns = ["latency"] + [col for col in df.columns if "score" in col]
        df_agg = df[agg_columns].mean().reset_index()
        df_agg.columns = ["metric", "value"]
        # Pivot the metric column to get a column for each metric and a row for each LLM app.
        df_agg = df_agg.pivot_table(index=None, columns="metric", values="value")
        # Drop the `metric` index.
        df_agg = df_agg.reset_index(drop=True)
        # Add the llm app config dict as columns in front of the aggregated results.
        df_agg = pd.concat([pd.json_normalize(llm_app.cfg), df_agg], axis=1)
        # Save the aggregated results as CSV.
        df_agg.to_csv(os.path.join(exp_dir, "results_agg.csv"), index=False)

        return df_agg
