from eval.eval import cli_evaluate
from fast_llm.models.ssm.external.eval.apriel_eval_wrapper import (  # noqa: F401
    AprielHybrid15bSSMWrapper,
    AprielHybridSSMWrapper,
    AprielSSMWrapper,
)

if __name__ == "__main__":
    cli_evaluate()
