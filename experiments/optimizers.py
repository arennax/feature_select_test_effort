import numpy as np
import random
from sklearn.tree import DecisionTreeRegressor
from experiments.useful_tools import KFold_df, normalize, mre_calc, sa_calc


def de(fun_opt, bounds, mut=0.8, crossp=0.7, popsize=20, itrs=10):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds)[:,0], np.asarray(bounds)[:,1]
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fun_opt(*ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(itrs):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fun_opt(*trial_denorm)
            if f < fitness[j]:  ####### MRE
                # if f > fitness[j]:  ####### SA
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:  ####### MRE
                    # if f > fitness[best_idx]:  ####### SA
                    best_idx = j
                    best = trial_denorm
        # yield best, fitness[best_idx]
        return fitness[best_idx]


def flash(train_input, train_actual_effort, test_input, test_actual_effort, pop_size):
    def convert(index):
        a = int(index / 240 + 1)
        b = int(index % 240 / 20 + 1)
        c = int(index % 20 + 2)
        return a, b, c

    all_case = set(range(0, 2880))
    modeling_pool = random.sample(all_case, pop_size)

    List_X = []
    List_Y = []

    for i in range(len(modeling_pool)):
        temp = convert(modeling_pool[i])
        List_X.append(temp)
        model = DecisionTreeRegressor(max_depth=temp[0], min_samples_leaf=temp[1], min_samples_split=temp[2])
        model.fit(train_input, train_actual_effort)
        test_predict_effort = model.predict(test_input)
        test_predict_Y = test_predict_effort
        test_actual_Y = test_actual_effort.values

        List_Y.append(mre_calc(test_predict_Y, test_actual_Y))  ######### for MRE flash
        # List_Y.append(sa_calc(test_predict_Y, test_actual_Y))  ######### for SA flash

    remain_pool = all_case - set(modeling_pool)
    test_list = []
    for i in list(remain_pool):
        test_list.append(convert(i))

    upper_model = DecisionTreeRegressor()
    life = 20

    while len(List_X) < 201 and life > 0:  # eval_number
        upper_model.fit(List_X, List_Y)
        candidate = random.sample(test_list, 1)
        test_list.remove(candidate[0])
        candi_pred_value = upper_model.predict(candidate)
        if candi_pred_value < np.median(List_Y):  ######### for MRE flash
            # if candi_pred_value > np.median(List_Y):  ######### for SA flash
            List_X.append(candidate[0])
            candi_config = candidate[0]
            candi_model = DecisionTreeRegressor(max_depth=candi_config[0], min_samples_leaf=candi_config[1],
                                                min_samples_split=candi_config[2])
            candi_model.fit(train_input, train_actual_effort)
            candi_pred_Y = candi_model.predict(test_input)
            candi_actual_Y = test_actual_effort.values

            List_Y.append(mre_calc(candi_pred_Y, candi_actual_Y))  ######### for MRE flash
            # List_Y.append(sa_calc(candi_pred_Y, candi_actual_Y))  ######### for SA flash

        else:
            life -= 1

    return np.min(List_Y)  ########## min for MRE
    # return np.max(List_Y)  ########## min for SA


if __name__ == '__main__':

    def adder(a,b):
        return a+b


    it = de(adder, bounds=[(0, 10), (0, 10)])
    print(it)
