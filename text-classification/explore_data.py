
def get_num_classes(labels):
    num_classes = max(labels) + 1
    missing_classes = [i for i in range(num_classes) if i not in labels]
    
    if len(missing_classes) > 0:
        raise ValueError('Missing samples with label value(s) '
                         '{missing_classes}. Please make sure you have '
                         'at least one sample for every label value '
                         'in the range(0, {max_class})'.format(
                            missing_classes=missing_classes,
                            max_class=num_classes - 1))
    if num_classes <= 1:
        raise ValueError('Invalid number of labels: {num_classes}. '
                         'Please make sure there are at least two classes '
                         'of samples'.format(num_classes=num_classes))
    return num_classes