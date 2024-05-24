import pytest

from multiSero.array_analyzer.extract.metadata import MetaData
import multiSero.array_analyzer.extract.constants as constants

"""
Tests to add:
METADATA:
normal operation:
*requires:
    - input folder with good xlsx, good 
    - dummy output folder
1) extension .xml, .xlsx are found

2) no extension is found (not implemented error)

3) c.params are populated correctly for each of
    3a) .xml, .xlsx
    
4) arrays are populated correctly for each of 
    4a) .xml, .xlsx
    
5) fiducials are calculated correctly for each of
    5a) 6x6 array metadata, .xml and .xlsx
    5b) 6x8 array metadata, .xml and .xlsx
    
bad operation:
* requires:
    - input folder with bad xml:
        - multiple .xml
        - .xml with correct dictionaries
1) catch when multiple .xml
2) catch when incorrectly labeled .xlsx
3)      txtparser xlsx: catch when incorrect sheets
2) c.params has rows, columns, pitch, size defined
3) 
"""


def test_one_xml(create_good_xml):
    input_dir, output_dir = create_good_xml
    constants.METADATA_FILE = 'temp.xml'
    constants.RUN_PATH = output_dir
    MetaData(input_dir, output_dir)


def test_wrong_xml_name(create_good_xml):
    input_dir, output_dir = create_good_xml
    constants.METADATA_FILE = 'no_xml.xml'
    constants.RUN_PATH = output_dir
    with pytest.raises(IOError):
        MetaData(input_dir, output_dir)


def test_xlsx(create_good_xlsx):
    input_dir, output_dir = create_good_xlsx
    constants.METADATA_FILE = 'multisero_output_data_metadata.xlsx'
    constants.RUN_PATH = output_dir
    MetaData(input_dir, output_dir)
    assert constants.params['rows'] == 6
    assert constants.params['columns'] == 6
    assert constants.params['v_pitch'] == 0.4
    assert constants.params['h_pitch'] == 0.45
    assert constants.params['spot_width'] == 0.2
    assert constants.params['pixel_size'] == 0.0049


def test_xlsx_rerun(create_good_xlsx):
    input_dir, output_dir = create_good_xlsx
    constants.METADATA_FILE = 'multisero_output_data_metadata.xlsx'
    constants.RUN_PATH = output_dir
    constants.RERUN = True
    MetaData(input_dir, output_dir)
    assert constants.RERUN_WELLS == ['A3', 'B7']
    assert constants.params['rows'] == 6
    assert constants.params['columns'] == 6
    assert constants.params['v_pitch'] == 0.4
    assert constants.params['h_pitch'] == 0.45
    assert constants.params['spot_width'] == 0.2
    assert constants.params['pixel_size'] == 0.0049
