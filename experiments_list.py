experiments_title = {
    "no patterns": "",
    # Rules identified by uncertainty sampling
    "pattern first": "OBJ_RIGHT_PERSON",
    "pattern second": "SUBJ-PERSON , OBJ-TITLE",
    "pattern third": "SUBJ-PERSON , DT OBJ-TITLE",
    "pattern fourth": "SUBJ-PERSON , DT JJ OBJ-TITLE",
    "pattern fifth": "OBJ_LEFT_LOCATION",
    "top four patterns": "OBJ_RIGHT_PERSON;SUBJ-PERSON , OBJ-TITLE;SUBJ-PERSON , DT OBJ-TITLE;SUBJ-PERSON , DT JJ OBJ-TITLE",
    "pattern negative first": "OBJ-TITLE SUBJ-PERSON",
    "pattern negative second": "OBJ_LEFT_ORGANIZATION",
    "pattern negative third": "OBJ_LEFT_MISC",
    "pattern negative fourth": "OBJ-TITLE , SUBJ-PERSON",
    "pattern combination first": "OBJ_RIGHT_PERSON;OBJ-TITLE SUBJ-PERSON"
    "pattern combination second": "OBJ_RIGHT_PERSON;OBJ-TITLE SUBJ-PERSON;SUBJ-PERSON , OBJ-TITLE;SUBJ-PERSON , DT OBJ-TITLE;SUBJ-PERSON , DT JJ OBJ-TITLE",
    "pattern combination third": "OBJ_RIGHT_PERSON;OBJ-TITLE SUBJ-PERSON;SUBJ-PERSON , OBJ-TITLE;SUBJ-PERSON , DT OBJ-TITLE"
    "pattern combination fourth": "OBJ_RIGHT_PERSON;OBJ-TITLE SUBJ-PERSON;SUBJ-PERSON , OBJ-TITLE;SUBJ-PERSON , DT OBJ-TITLE;SUBJ-PERSON , DT JJ OBJ-TITLE;OBJ-TITLE , SUBJ-PERSON"
    "pattern combination best": "OBJ_RIGHT_PERSON;OBJ-TITLE SUBJ-PERSON;SUBJ-PERSON , DT OBJ-TITLE;SUBJ-PERSON , DT JJ OBJ-TITLE"
    "pattern combination filtered": "OBJ-TITLE SUBJ-PERSON;OBJ_RIGHT_PERSON;SUBJ-PERSON , OBJ-TITLE;SUBJ-PERSON , DT OBJ-TITLE"
}


experiments_employee = {
    "no patterns": "",
    # Rules identified by uncertainty sampling
    "pattern first": "SUBJ-PERSON , NN IN DT OBJ-ORGANIZATION",
    "pattern second": "OBJ-ORGANIZATION spokesman SUBJ-PERSON",
    "pattern third": "OBJ-ORGANIZATION , SUBJ-PERSON",
    "pattern fourth": "SUBJ-PERSON , OBJ-ORGANIZATION",
    "pattern negative first": "OBJ-ORGANIZATION NN SUBJ-PERSON",
    "pattern combination second": "SUBJ-PERSON , NN IN DT OBJ-ORGANIZATION;OBJ-ORGANIZATION spokesman SUBJ-PERSON;OBJ-ORGANIZATION , SUBJ-PERSON;SUBJ-PERSON , OBJ-ORGANIZATION",
    "pattern combination first": "SUBJ-PERSON , NN IN DT OBJ-ORGANIZATION;OBJ-ORGANIZATION NN SUBJ-PERSON;OBJ-ORGANIZATION , SUBJ-PERSON;SUBJ-PERSON , OBJ-ORGANIZATION"
}


experiments_top_members = {
    "no patterns": "",
    # Rules identified by uncertainty sampling
    "pattern first": "OBJ-PERSON , JJ NN IN DT SUBJ-ORGANIZATION",
    "pattern second": "OBJ-PERSON , DT SUBJ-ORGANIZATION",
    "pattern third": "OBJ-PERSON VBD DT SUBJ-ORGANIZATION",
    "pattern fourth": "SUBJ-ORGANIZATION NNP OBJ-PERSON",
    "pattern fifth": "OBJ-PERSON , executive director of the SUBJ-ORGANIZATION",
    "pattern negative first": "SUBJ-ORGANIZATION NN OBJ-PERSON"
    "top three patterns": "OBJ-PERSON , JJ NN IN DT SUBJ-ORGANIZATION;OBJ-PERSON , DT SUBJ-ORGANIZATION;OBJ-PERSON VBD DT SUBJ-ORGANIZATION"
    "pattern combination first": "OBJ-PERSON , JJ NN IN DT SUBJ-ORGANIZATION;OBJ-PERSON VBD DT SUBJ-ORGANIZATION"
    "pattern combination second": "OBJ-PERSON , JJ NN IN DT SUBJ-ORGANIZATION;SUBJ-ORGANIZATION NNP OBJ-PERSON",
    "pattern combination third": "OBJ-PERSON , JJ NN IN DT SUBJ-ORGANIZATION;SUBJ-ORGANIZATION NNP OBJ-PERSON;OBJ-PERSON VBD DT SUBJ-ORGANIZATION"
}

