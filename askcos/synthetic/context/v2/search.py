import numpy as np
import tensorflow as tf

def remove_duplicates_results(results):
    '''
        res = [(reagents_onehot, score)]
        The highest score is kept
    '''
    tmp = {}
    for r in results:
        key = tuple(r[0].astype('int32').flatten().tolist())
        val = r[1]
        tmp[key] = max(tmp.get(key, -np.inf), val)
    res = []
    for k, v in tmp.items():
        res.append((np.array(k, dtype='float32'),v))
    return res


def top_n_results(_res, n=10, remove_duplicates=True):
    '''
        res = [(reagents_onehot, score)]
    '''
    if remove_duplicates:
        res = remove_duplicates_results(_res)
    scores = np.array([i[1] for i in res], dtype='float32')
    idx    = list(np.flip(np.argsort(scores)))
    if len(idx) <= n:
        return [res[i] for i in idx]
    else:
        return [res[i] for i in idx[0:n]]


def min_score_results(res):
    '''
        res = [(reagents_onehot, score)]
    '''
    scores = np.array([i[1] for i in res], dtype='float32')
    return np.min(scores)


def max_score_results(res):
    '''
        res = [(reagents_onehot, score)]
    '''
    scores = np.array([i[1] for i in res], dtype='float32')
    return np.max(scores)
    

def is_target_in_results(res, tgt):
    cnt = 0
    for i in res:
        if np.all(np.abs(i[0] - tgt) < 1e-6):
            return True, cnt
        cnt += 1
    return False, cnt


def score_geometry_average(results):
    res = []
    for r in results:
        score = r[1]
        n = np.sum(r[0])+1
        score = score**(1.0/n)
        res.append((r[0], score))
    return res


def beam_search(model, model_inputs, vacab_size, max_steps=8, beam_size=10, eos_id=0, keepall=False, returnall=True, reagents=None):
    '''
        eos_id: end of sequence logit
        keepall: keep all top beam_size during search, beam_size^Nstep
        returnall: no top_results
    '''
    res = [] # [(reagents_onehot, score)]
    if reagents is None:
        input_reagent = [(np.zeros(shape=(vacab_size), dtype='float32'), 1.0)]
    else:
        input_reagent = [(reagents, 1.0)]
    nstep = 0
    while len(input_reagent) != 0 and nstep < max_steps:
        new_input_reagent = []
        for r, score in input_reagent:
            model_inputs['Input_reagents'] = tf.reshape(tf.convert_to_tensor(r, dtype=tf.float32), shape=(1,vacab_size))
            y_pred = model(**model_inputs)['softmax_1'].numpy()
            y_pred_class_num = np.flip(np.argsort(y_pred, axis=-1), axis=-1)[0,0:beam_size]
            y_pred_class_score = y_pred[0, y_pred_class_num]
            for i in range(len(y_pred_class_num)):
                n = y_pred_class_num[i]
                s = y_pred_class_score[i]*score
                new_r = np.copy(r).flatten()
                if n == eos_id:
                    # move finished
                    res.append((new_r, s))
                else:
                    # set predicted reagents
                    new_r[n] = 1
                    new_input_reagent.append((new_r, s))
        if keepall:
            # remove duplicates, keep max score
            input_reagent = remove_duplicates_results(new_input_reagent)
        else:
            # keep beam_size only
            input_reagent = top_n_results(new_input_reagent, n=beam_size, remove_duplicates=True)
        # early termination, assume score < 1
        if len(res) > beam_size:
            res_top = top_n_results(res, n=beam_size, remove_duplicates=True)
            res_min = min_score_results(res_top)
            s_max = max_score_results(input_reagent)
            if s_max < res_min:
                break
        # increase step count
        nstep += 1
    if returnall:
        res = top_n_results(res, n=len(res), remove_duplicates=True)
    else:
        res = top_n_results(res, n=beam_size, remove_duplicates=True)
    return res


if __name__ == "__main__":
    a = [(np.array([1,1]), 0.5),(np.array([1,1]), 0.3),(np.array([1,0]), 0.5),(np.array([1,0]), 0.7)]
    r = remove_duplicates_results(a)
    assert '[(array([1., 1.]), 0.5), (array([1., 0.]), 0.7)]' == str(r)
