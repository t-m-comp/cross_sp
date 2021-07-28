#analysis result directory
analysis_directory = "analysis_set"

setting_feature_name_list = ["calc_values(tfdata,'speed',50,'variance')","calc_values(tfdata,'speed',5,'average')"]#

def check_normal_dop(class_name):

    if class_name=="human_Co" or class_name=="normal-pre" or class_name=="trans-Short_A" or class_name=="trans-norm" or class_name=="normal-pre_sm2":
        return "normal"
    elif class_name=="human_Pt" or class_name=="pd-pre" or class_name=="trans-longA" or class_name=="trans-dop3" or class_name=="pd-pre_sm2":
        return "dop"
    else:
        return None


def check_domain(class_name):

    if class_name=="human_Co" or class_name=="human_Pt":
        return "human"
    elif class_name=="normal-pre" or class_name=="pd-pre":
        return "mouse"
    elif class_name=="normal-pre_sm2" or class_name=="pd-pre_sm2":
        return "mouse_sm2"
    elif class_name=="trans-Short_A" or class_name=="trans-longA":
        return "kokunusuto"
    elif class_name=="trans-norm" or class_name=="trans-dop3":
        return "senchu"
    else:
        return None

def return_class_list(domain_name):

    if domain_name=="human":
        return ["human_Co","human_Pt"]
    elif domain_name=="mouse":
        return ["normal-pre","pd-pre"]
    elif domain_name=="mouse_sm2":
        return ["normal-pre_sm2","pd-pre_sm2"]
    elif domain_name=="kokunusuto":
        return ["trans-Short_A","trans-longA"]
    elif domain_name=="senchu":
        return ["trans-norm","trans-dop3"]
    else:
        return None
