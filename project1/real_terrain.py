from LinearRegression import *

data1 = 'data/SRTM_data_Norway_1.tif'
data2 = 'data/SRTM_data_Norway_2.tif'
order = 5
LR_terrain = LinearRegression(order=order, data=data1, reduce_factor=10, x_pos=0, y_pos=1950, scale=True)
LR_terrain.plot_terrain()
LR_terrain.plot_terrain_3D()
""" same analysis as in b) """
LR_terrain.execute_regression(method=LR_terrain.ols)
poly_degrees = np.arange(1, order+1)
plt.figure()
plt.plot(poly_degrees, LR_terrain.MSE_test, label='MSE test', linestyle="--")
plt.plot(poly_degrees, LR_terrain.MSE_train, label='MSE train')
plt.xlabel('Polynomial degree')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.tight_layout()
plt.savefig("figures/Terrain/terra_OLS_MSE.pdf")
plt.show()
plt.plot(poly_degrees, LR_terrain.R2_test, label=r'R$^2$ test', linestyle="--")
plt.plot(poly_degrees, LR_terrain.R2_train, label=r'R$^2$ train')
plt.xlabel('Polynomial degree')
plt.ylabel(r'$R^2$ score')
plt.legend()
plt.tight_layout()
plt.savefig("figures/Terrain/terra_OLS_R2.pdf")
plt.show()
betas = LR_terrain.beta
var = LR_terrain.var[::-1]
plt.figure()
# # print(np.shape(LR_b.var))
# # print(LR_b.var[-1])
ax = plt.axes()
color = plt.cm.viridis(np.linspace(0.9, 0,11))
ax.set_prop_cycle(plt.cycler('color', color))
ax.set_xticks([i for i in range(1, len(betas[-1])+1)])
for i, beta in enumerate(betas[::-1]):
    coefficients = beta[~(np.isnan(beta))]
    beta_indexes = np.arange(1, len(coefficients)+1)
    plt.errorbar(beta_indexes, coefficients, yerr=np.sqrt(var[i]), marker='o', linestyle='--', capsize=2, label='d = %d' % (5-i))
""" We need to figure out what to do with the scaling.
    Decide wether np.sqrt of variance as error bars or just the variance. 
        Also decide wether we have split's random_state be the same as random seed or set it 
        as something else as we have so far """
plt.xlabel(r'$\beta$ coefficient number')
plt.ylabel(r'$\beta$ coefficient value')
plt.legend()
plt.tight_layout()
plt.savefig("figures/Terrain/terra_OLS_beta.pdf")
plt.show()

""" same analysis as in c) BVT """
order = 12
LR_terrain = LinearRegression(order=order, data=data1, reduce_factor=10, x_pos=0, y_pos=1950, scale=True)
LR_terrain.execute_regression(method=LR_terrain.ols, bootstrap=True, n=300)

poly_degrees = np.arange(1, order+1)
plt.plot(poly_degrees, LR_terrain.MSE_train, label='train', color='orange', linestyle='--')
plt.plot(poly_degrees, LR_terrain.MSE_test,  label='test',  color='orange')
plt.legend()
plt.xlabel("Polynomial degree")
plt.ylabel("MSE score")
plt.tight_layout()
plt.show()
plt.savefig("figures/Terrain/terra_OLS_bootstrap.pdf")

plt.plot(poly_degrees, LR_terrain.BIAS,     label=r'BIAS$^2$',     color='red')
plt.plot(poly_degrees, LR_terrain.MSE_test, label='MSE test', color='orange')
plt.plot(poly_degrees, LR_terrain.var,      label='var',      color='green')   
plt.legend()
plt.xlabel("Polynomial degree")
plt.ylabel("score")
plt.tight_layout()
plt.show()
plt.savefig("figures/Terrain/terra_OLS_biasvar.pdf")

""" same cross validation analysis as in d) with OLS """
""" NB!! VERY ODD VALUES """
kfolds = [i for i in range(5, 11)]
LR_terrain.execute_regression(method=LR_terrain.ols, bootstrap=True, n=300)
LR_terrain.execute_regression(method=LR_terrain.ols, crossval=True, kfolds=kfolds)
poly_degrees = np.arange(1, order+1)
fig, ax = plt.subplots()
plt.plot(poly_degrees, LR_terrain.MSE_train, label='bootstrap train', color='k', linestyle='--')
plt.plot(poly_degrees, LR_terrain.MSE_test,  label='bootstrap test', color='k')
color = plt.cm.cool(np.linspace(0.9, 0,11))
ax.set_prop_cycle(plt.cycler('color', color))
for k in range(len(kfolds)):
    plt.plot(poly_degrees, LR_terrain.MSE_CV[k], label=f'k = {kfolds[k]}')
plt.legend(loc='upper center')
plt.xlabel("Polynomial degree")
plt.ylabel("MSE score")
plt.tight_layout()
plt.savefig("figures/Terrain/terra_OLS_crossval.pdf")
plt.show()


""" e) """
""" Ridge Bootstrap """
order = 12
poly_degrees = np.arange(1, order+1)
hyperparams = [10**i for i in range(-10, 0)]
extent = [poly_degrees[0], poly_degrees[-1], hyperparams[0], hyperparams[-1]]
LR_terrain.execute_regression(method=LR_terrain.ridge, bootstrap=True, n=100, hyperparams=hyperparams)
MSE_ridge_bootstrap = LR_terrain.MSE_bootstrap

fig, ax = plt.subplots()
plt.contourf(MSE_ridge_bootstrap, extent=extent, levels=30)
plt.xlabel("Polynomial degree")
plt.ylabel("Pentalty parameter")
cbar = plt.colorbar(pad=0.01)
cbar.set_label('MSE score')
plt.tight_layout()
plt.savefig("figures/Terrain/terra_Ridge_bootstrap.pdf")
plt.show()

""" Ridge bias-var analysis analysis with bootstrap """
BIAS_ridge_bootstrap = LR_terrain.BIAS_bootstrap
var_ridge_bootstrap = LR_terrain.var_bootstrap
for k in range(len(hyperparams)):
	h1 = plt.plot(poly_degrees, MSE_ridge_bootstrap[k],  label='MSE test',  color='orange', alpha=k*0.1)#, s=15)
	h2 = plt.plot(poly_degrees, BIAS_ridge_bootstrap[k], label=r'BIAS$^2$', color='blue',   alpha=k*0.1)#, s=15)
	h3 = plt.plot(poly_degrees, var_ridge_bootstrap[k],  label='var',       color='red',    alpha=k*0.1)#, s=15)
	plt.legend(handles=[h1[0], h2[0], h3[0]])#labels=["MSE_test", "BIAS", "Variance"])
plt.xlabel("Polynomial degree")
plt.ylabel("MSE score")
plt.tight_layout()
plt.savefig("figures/Terrain/terra_Ridge_biasvar.pdf")
plt.show()

""" Ridge Cross Val """
""" e) heatmap w. cross validation yields crazy values with real data """
kfolds = [i for i in range(5, 11)]
LR_terrain.execute_regression(method=LR_terrain.ridge, crossval=True, kfolds=10, hyperparams=hyperparams)
MSE_ridge_crossval = LR_terrain.MSE_crossval
# min_MSE_idx = divmod(MSE_ridge_crossval.argmin(), MSE_ridge_crossval.shape[1])
fig, ax = plt.subplots()
plt.contourf(MSE_ridge_crossval, extent=extent, levels=30)
# sns.heatmap(MSE_ridge_crossval.T, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'Accuracy'},fmt='.1e')
# ax.add_patch(plt.Rectangle((min_MSE_idx[0], min_MSE_idx[1]), 1, 1, fc='none', ec='red', lw=2, clip_on=False))
plt.xlabel("Polynomial degree")
plt.ylabel("Pentalty parameter")
cbar = plt.colorbar(pad=0.01)
cbar.set_label('MSE score')
plt.tight_layout()
plt.savefig("figures/Terrain/terra_Ridge_crossval.pdf")
plt.show()


""" Task f) """
""" Lasso Bootstrap """
""" Lasso does not converge when scale=True, hints that scaling is not implemented correctly """

order = 12
poly_degrees = np.arange(1, order+1)
hyperparams = [10**i for i in range(-10, 0)]
extent = [poly_degrees[0], poly_degrees[-1], np.log10(hyperparams[0]), np.log10(hyperparams[-1])]

LR_terrain = LinearRegression(order=order, data=data1, reduce_factor=10, x_pos=0, y_pos=1950, scale=True)
LR_terrain.execute_regression(method=LR_terrain.lasso, bootstrap=True, n=10, hyperparams=hyperparams)
MSE_lasso_bootstrap = LR_terrain.MSE_bootstrap


# """ Lasso heatmap boot """
# fig, ax = plt.subplots()
# plt.contourf(MSE_lasso_bootstrap, extent=extent, levels=30)#(order*len(hyperparams)))
# # plt.contourf(poly_degrees, hyperparams, MSE_ridge_bootstrap, cmap=plt.cm.magma, levels=30)
# # plt.plot(min_MSE_idx[0], min_MSE_idx[1], 'o', color='red')
# plt.xlabel("Polynomial degree")
# plt.ylabel(r"Penalty parameter [log$_{10}$]")
# cbar = plt.colorbar(pad=0.01)
# cbar.set_label('MSE score')
# plt.tight_layout()
# plt.savefig("figures/Terrain/terra_Lasso_bootstrap.pdf")
# plt.show()

# """ Lasso heatmap Cross Validation"""
# kfolds = [i for i in range(5, 11)]
# LR_terrain.execute_regression(method=LR_terrain.lasso, crossval=True, kfolds=10, hyperparams=hyperparams)
# MSE_lasso_crossval = LR_terrain.MSE_crossval
# fig, ax = plt.subplots()
# plt.contourf(MSE_lasso_crossval, extent=extent, levels=30)
# plt.xlabel("Polynomial degree")
# plt.ylabel(r"Penalty parameter [log$_{10}$]")
# cbar = plt.colorbar(pad=0.01)
# cbar.set_label('MSE score')
# plt.tight_layout()
# plt.savefig("figures/terrain/terra_Lasso_crossval.pdf")
# plt.show()

