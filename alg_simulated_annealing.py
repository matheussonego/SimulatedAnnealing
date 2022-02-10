#!/usr/bin/python3
# -*- coding: utf-8 -*-

DEFAULT_OUT = "alg_simulated_annealing.txt"
DEFAULT_SEED = None

DEFAULT_N_START = 1
DEFAULT_N_STEP = 1
DEFAULT_TRIALS = 3
DEFAULT_TEMPERATURA_MINIMA = 0.001
DEFAULT_ALPHA = 0.9
DEFAULT_REPETICOES = 500

from subprocess import Popen, PIPE
from time import sleep, time
from multiprocessing import Process
import shlex
import json

import sys
import os
import argparse
import logging
import subprocess
import random
import decimal
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import timeit

f = open("bairros.txt","r")
l_bairros = f.read().split()
f.close()
bairros = {} #dicionario com a posição de todas os bairros {'bairro_number':[posX, posY]}

for x in range(0,len(l_bairros),3): #preenche o dicionario com os bairros e posições
    bairros[l_bairros[x]] = [float(l_bairros[x+1]),float(l_bairros[x+2])]

def distancia(xyA,xyB): #calcula o tamanho da reta entre dois pontos
    xA, xB, yA, yB = (xyA[0]), (xyB[0]), (xyA[1]), (xyB[1])
    d = sqrt((xB-xA)**2 + (yB-yA)**2)
    return round(d,12)

TAMANHO = len(bairros)
bairros_custo = {} #dicionario com o custo de cada viagem {('bairroA_number','bairroB_number'): distancia}
for k in range(1,TAMANHO + 1):
    for c in range(1,TAMANHO + 1):
        bairros_custo[(str(k),str(c))] = distancia(bairros[str(k)],bairros[str(c)])

def custo_total(lista_bairros): #retorna o custo total de uma solução
    custo = 0
    for bairro in range(len(lista_bairros)):
        if bairro == len(lista_bairros)-1: #se chegou no último bairro soma o custo com a origem
            custo += bairros_custo[(str(lista_bairros[bairro]), str(lista_bairros[0]))]
        else:
            custo+=bairros_custo[(str(lista_bairros[bairro]),str(lista_bairros[bairro+1]))]
    return custo

def vizinho(solucao):
    solucao_anterior = solucao.copy()
    nmax = len(solucao) - 1
    while True:
        posA = random.randint(0,nmax)
        posB = random.randint(0,nmax)
        a = solucao[posA]
        b = solucao[posB]
        solucao[posA] = b
        solucao[posB] = a
        if solucao != solucao_anterior:
            break
    return solucao

def probabilidade(custo_antigo,custo_novo,temperatura): #calcula a probabilidade de aceitação da nova solução
    decimal.getcontext().prec = 100
    diferenca_custo = custo_antigo - custo_novo
    custo_temp = diferenca_custo/temperatura
    p = decimal.Decimal(0)
    e = decimal.Decimal(2.71828)
    n_custo_temp = decimal.Decimal(-custo_temp)
    try:
        p = e**n_custo_temp
        resultado = repr(p)
    except decimal.Overflow:
        #print("Error decimal Overflow")
        return 0.0

    try: #caso o número tenha casas decimais
        fim = resultado.find("')")
        resultado = round(float(resultado[9:fim-1]), 3)

    except: #número n tem casas decimais
        resultado = round(float(resultado[9:-2]))
    return resultado

def annealing(solution, T_min, alpha, repetitions):
    old_cost = custo_total(solution)
    T = 1.0
    best_solution, best_cost = solution[::], old_cost
    while T > T_min:
        i = 1
        while i <= repetitions:
            new_solution = vizinho(solution)
            new_cost = custo_total(new_solution)
            p = probabilidade(old_cost, new_cost, T)
            if new_cost < best_cost:
                best_solution = new_solution[::]
                best_cost = new_cost
            if p > round(random.random(), 3):
                solution = new_solution[::]
                old_cost = new_cost
            i += 1

        T = T*alpha
    return best_solution, best_cost

def gerar_solucao(tamanho): #gera uma solução aleatória
    solucao_aleatoria = [x for x in range(1,tamanho + 1)]
    random.shuffle(solucao_aleatoria)
    return solucao_aleatoria

def main():
	# Definição de argumentos
	parser = argparse.ArgumentParser(description='Naive TPS')
	help_msg = "arquivo de saída.  Padrão:{}".format(DEFAULT_OUT)
	parser.add_argument("--out", "-o", help=help_msg, default=DEFAULT_OUT, type=str)

	help_msg = "semente aleatória. Padrão:{}".format(DEFAULT_SEED)
	parser.add_argument("--seed", "-s", help=help_msg, default=DEFAULT_SEED, type=int)

	help_msg = "n máximo.          Padrão:{}".format(TAMANHO)
	parser.add_argument("--nstop", "-n", help=help_msg, default=TAMANHO, type=int)

	help_msg = "n mínimo.          Padrão:{}".format(DEFAULT_N_START)
	parser.add_argument("--nstart", "-a", help=help_msg, default=DEFAULT_N_START, type=int)

	help_msg = "n passo.           Padrão:{}".format(DEFAULT_N_STEP)
	parser.add_argument("--nstep", "-e", help=help_msg, default=DEFAULT_N_STEP, type=int)

	help_msg = "tentativas.        Padrão:{}".format(DEFAULT_N_STEP)
	parser.add_argument("--trials", "-t", help=help_msg, default=DEFAULT_TRIALS, type=int)

	help_msg = "temperatura mínima.         Padrão:{}".format(DEFAULT_TEMPERATURA_MINIMA)
	parser.add_argument("--tempmin", "-tm", help=help_msg, default=DEFAULT_TEMPERATURA_MINIMA, type=float)

	help_msg = "alpha.        Padrão:{}".format(DEFAULT_ALPHA)
	parser.add_argument("--alpha", "-p", help=help_msg, default=DEFAULT_ALPHA, type=float)

	help_mesg = "repetições.         Padrão:{}".format(DEFAULT_REPETICOES)
	parser.add_argument("--repetitions", "-r", help=help_msg, default=DEFAULT_REPETICOES, type=float)

	# Lê argumentos from da linha de comando
	args = parser.parse_args()


	trials = args.trials
	f = open(args.out, "w")
	f.write("#Selection sort\n")
	f.write("#n time_s_avg time_s_std (for {} trials)\n".format(trials))
	np.random.seed(args.seed)
	for n in range(args.nstart, args.nstop+1, args.nstep): #range(1, 100):
		resultados = [0 for i in range(trials)]
		tempos = [0 for i in range(trials)]
		for trial in range(trials):
			print("\n-------")
			print("n: {} trial: {}".format(n, trial+1))
			entrada = gerar_solucao(n)
			print("Entrada: {}".format(entrada))
			tempo_inicio = timeit.default_timer()
			resultados[trial] = annealing(entrada, args.tempmin, args.alpha, args.repetitions)
			tempo_fim = timeit.default_timer()
			tempos[trial] = tempo_fim - tempo_inicio
			print("Saída: {}".format(resultados[trial]))
			print('Tempo: {} s'.format(tempos[trial]))
			print("")

		tempos_avg = np.average(tempos)  # calcula média
		tempos_std = np.std(a=tempos, ddof=False)  # ddof=calcula desvio padrao de uma amostra?


		f.write("{} {} {}\n".format(n, tempos_avg, tempos_std))
	f.close()


if __name__ == '__main__':
	sys.exit(main())