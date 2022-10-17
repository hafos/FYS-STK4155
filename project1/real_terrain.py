from LinearRegression import *

data1 = 'data/SRTM_data_Norway_1.tif'
data2 = 'data/SRTM_data_Norway_2.tif'
order = 5
LR_terrain = LinearRegression(order=order, data=data1, reduce_factor=10, x_pos=0, y_pos=1950, scale=True)
# LR_terrain.plot_terrain()
# LR_terrain.plot_terrain_3D()
""" same analysis as in b) """
# LR_terrain.execute_regression(method=LR_terrain.ols)
# poly_degrees = np.arange(1, order+1)
# plt.figure()
# plt.plot(poly_degrees, LR_terrain.MSE_test, label='MSE test')
# plt.plot(poly_degrees, LR_terrain.MSE_train, label='MSE train')
# plt.xlabel('Polynomial degree')
# plt.ylabel('Mean Squared Error')
# plt.legend(); plt.show()
# plt.plot(poly_degrees, LR_terrain.R2_test, label=r'$R^2 test$')
# plt.plot(poly_degrees, LR_terrain.R2_train, label=r'$R^2 train$')
# plt.xlabel('Polynomial degree')
# plt.ylabel(r'$R^2 \; Score$')
# plt.legend(); plt.show()
# betas = LR_terrain.beta
# var = LR_terrain.var[::-1]
# plt.figure()
# # # print(np.shape(LR_b.var))
# # # print(LR_b.var[-1])
# ax = plt.axes()
# color = plt.cm.viridis(np.linspace(0.9, 0,11))
# ax.set_prop_cycle(plt.cycler('color', color))#["axes.prop_cycle"] = plt.cycler('color', color)
# ax.set_xticks([i for i in range(1, len(betas[-1])+1)])
# for i, beta in enumerate(betas[::-1]):
#     coefficients = beta[~(np.isnan(beta))]
#     beta_indexes = np.arange(1, len(coefficients)+1)
#     plt.errorbar(beta_indexes, coefficients, yerr=np.sqrt(var[i]), marker='o', linestyle='--', capsize=2, label='d = %d' % (5-i))
# """ We need to figure out what to do with the scaling.
#     Decide wether np.sqrt of variance as error bars or just the variance. 
#         Also decide wether we have split's random_state be the same as random seed or set it 
#         as something else as we have so far """
# plt.xlabel('Order of polynomial')
# plt.ylabel(r'$\beta\; coefficient \;value$')
# plt.legend()
# plt.show()

""" same analysis as in c) BVT """
order = 12
# LR_terrain = LinearRegression(order=order, data=data1, reduce_factor=10, x_pos=0, y_pos=1950, scale=True)
# LR_terrain.execute_regression(method=LR_terrain.ols, bootstrap=True, n=400)
# poly_degrees = np.arange(1, order+1)
# plt.figure()
# plt.plot(poly_degrees, LR_terrain.MSE_train, label='train')
# plt.plot(poly_degrees, LR_terrain.MSE_test, label='test')
# plt.legend()
# plt.show()
# plt.figure()
# plt.plot(poly_degrees, LR_terrain.BIAS, label='BIAS', color='orange')#, s=15)
# plt.plot(poly_degrees, LR_terrain.MSE_test, label='MSE_test', color='blue')#, s=15)
# plt.plot(poly_degrees, LR_terrain.var, label='var', color='red')#, s=15)     
# plt.legend()
# plt.show()

""" same cross validation analysis as in d) with OLS """
order = 12
LR_terrain = LinearRegression(order=order, data=data1, reduce_factor=10, x_pos=0, y_pos=1950, scale=False)
kfolds = [i for i in range(5, 11)]
# LR_terrain.execute_regression(method=LR_terrain.ols, bootstrap=True, n=400)
LR_terrain.execute_regression(method=LR_terrain.ols, crossval=True, kfolds=kfolds)
poly_degrees = np.arange(1, order+1)
fig, ax = plt.subplots()
# plt.plot(poly_degrees, LR_terrain.MSE_train, label='bootstrap train')
# plt.plot(poly_degrees, LR_terrain.MSE_test, label='bootstrap test', color='k')
color = plt.cm.cool(np.linspace(0.9, 0,11))
ax.set_prop_cycle(plt.cycler('color', color))
for k in range(len(kfolds)):
    plt.plot(poly_degrees, LR_terrain.MSE_CV[k], label=f'crossval k: {kfolds[k]}')
plt.legend(fontsize=14)
plt.show()