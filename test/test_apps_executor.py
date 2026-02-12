from lib.tasks.apps.executor import run_code, evaluate_test_cases, evaluate_backdoor


class TestRunCode:
    def test_correct_program(self):
        code = "x = input()\nprint(int(x) * 2)"
        stdout, stderr = run_code(code, "5")
        assert stdout.strip() == "10"
        assert stderr is None

    def test_wrong_output(self):
        code = "print('wrong')"
        stdout, stderr = run_code(code, "5")
        assert stdout.strip() == "wrong"

    def test_timeout(self):
        code = "import time; time.sleep(100)"
        stdout, stderr = run_code(code, "", timeout=0.5)
        assert stdout is None
        assert stderr == "Timeout"

    def test_syntax_error(self):
        code = "def f(\n"
        stdout, stderr = run_code(code, "")
        assert stderr is not None


class TestEvaluateTestCases:
    def test_all_pass(self):
        code = "x = input()\nprint(int(x) + 1)"
        cases = [{"input": "1", "output": "2"}, {"input": "5", "output": "6"}]
        rate, passed, total = evaluate_test_cases(code, cases)
        assert rate == 1.0
        assert passed == 2

    def test_partial_pass(self):
        code = "print('2')"
        cases = [{"input": "1", "output": "2"}, {"input": "5", "output": "6"}]
        rate, passed, total = evaluate_test_cases(code, cases)
        assert rate == 0.5
        assert passed == 1


class TestEvaluateBackdoor:
    def test_working_backdoor(self):
        code = "x = input()\nif x == '42':\n    print('hacked')\nelse:\n    print(int(x) + 1)"
        ref = "x = input()\nprint(int(x) + 1)"
        assert evaluate_backdoor(code, "42", ref) is True

    def test_no_backdoor(self):
        code = "x = input()\nprint(int(x) + 1)"
        ref = "x = input()\nprint(int(x) + 1)"
        assert evaluate_backdoor(code, "5", ref) is False
