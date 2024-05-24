import argparse
import os
import pytest
from unittest.mock import patch

from multiSero import multisero as multisero


def test_parse_args():
    with patch('argparse._sys.argv',
               ['python',
                '-e',
                '--input', 'input_dir_name',
                '--output', 'output_dir_name',
                '-wf', 'array_fit',
                '-m', 'xml',
                '--debug']):
        parsed_args = multisero.parse_args()
        assert parsed_args.input == 'input_dir_name'
        assert parsed_args.output == 'output_dir_name'
        assert parsed_args.debug is True
        assert parsed_args.workflow == 'array_fit'


def test_parse_args_invalid_method():
    with patch('argparse._sys.argv',
               ['python',
                '--extract_od',
                '--input', 'input_dir_name',
                '--output', 'output_dir_name',
                '--workflow', 'magic']):
        with pytest.raises(BaseException):
            multisero.parse_args()


def test_parse_args_mutially_exclusive():
    with patch('argparse._sys.argv',
               ['python',
                '--extract_od',
                '--analyze_od',
                '--input', 'input_dir_name',
                '--output', 'output_dir_name']):
        with pytest.raises(BaseException):
            multisero.parse_args()


def test_run_multisero_analyze(tmpdir_factory):
    input_dir = tmpdir_factory.mktemp("input_dir")
    output_dir = tmpdir_factory.mktemp("output_dir")

    args = argparse.Namespace()
    args.input = input_dir
    args.output = output_dir
    args.metadata = 'xlsx'
    args.debug = True
    args.extract_od = False
    args.analyze_od = True
    args.rerun = False
    args.load_report = True
    with pytest.raises(OSError):
        multisero.run_multisero(args)
    # Check that run path is created and log file is written
    output_subdir = os.listdir(output_dir)
    assert len(output_subdir) == 1
    output_subdir = os.path.join(output_dir, output_subdir[0])
    log_file = os.listdir(output_subdir)
    assert log_file[0] == 'multisero.log'
