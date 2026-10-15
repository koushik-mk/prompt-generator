"""Shared test helpers: a fake Workers AI binding.

The fake binding records every invocation and can be configured to return
canned responses or raise exceptions, so no real Workers AI usage is
consumed by the test suite.
"""


class FakeAIBinding:
    """Mimics `env.AI.run(model_id, payload)` from the Workers AI binding."""

    def __init__(self, responses=None, errors=None):
        """
        responses: value (or list of values) returned by run().
        errors: exception (or list of exceptions) raised by run().
        When a list is given, items are consumed per call; the last item is
        reused for subsequent calls.
        """
        self.calls = []
        self._responses = responses
        self._errors = errors

    async def run(self, model_id, payload):
        self.calls.append({"model_id": model_id, "payload": payload})
        index = len(self.calls) - 1

        if self._errors is not None:
            errors = (
                self._errors if isinstance(self._errors, list) else [self._errors]
            )
            error = errors[min(index, len(errors) - 1)]
            if error is not None:
                raise error

        if self._responses is None:
            return {"response": "Role: fake generated prompt"}
        responses = (
            self._responses
            if isinstance(self._responses, list)
            else [self._responses]
        )
        return responses[min(index, len(responses) - 1)]

    @property
    def model_ids(self):
        return [call["model_id"] for call in self.calls]
