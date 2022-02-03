from math import sqrt
from matplotlib import pyplot as plt
import networkx as nw
import random, decimal, time, sys

f = open(sys.argv[1],"r")
l_bairros = f.read().split()
f.close()
bairros = {} #dicionario com a posição de todas os bairros {'bairro_number':[posX, posY]}

for x in range(0,len(l_bairros),3): #preenche o dicionario com os bairros e posições
    bairros[l_bairros[x]] = [float(l_bairros[x+1]),float(l_bairros[x+2])]

def distancia(xyA,xyB): #calcula o tamanho da reta entre dois pontos
    xA, xB, yA, yB = (xyA[0]), (xyB[0]), (xyA[1]), (xyB[1])
    d = sqrt((xB-xA)**2 + (yB-yA)**2)
    return round(d,12)

tamanho = len(bairros)
bairros_custo = {} #dicionario com o custo de cada viagem {('bairroA_number','bairroB_number'): distancia}
for k in range(1,tamanho + 1):
    for c in range(1,tamanho + 1):
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

        posC = random.randint(0, nmax)
        posD = random.randint(0, nmax)
        c = solucao[posC]
        d = solucao[posD]
        solucao[posC] = d
        solucao[posD] = c
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

repeticoes_de_mudanca = int(sys.argv[2])
def annealing(solution):
    old_cost = custo_total(solution)
    T = 1.0
    T_min = 0.0001
    alpha = 0.9
    best_solution, best_cost = solution[::], old_cost
    while T > T_min:
        i = 1
        while i <= repeticoes_de_mudanca:
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

def gerar_solucao(): #gera uma solução aleatória
    solucao_aleatoria = [x for x in range(1,tamanho + 1)]
    random.shuffle(solucao_aleatoria)
    return solucao_aleatoria


solucao_inicial = gerar_solucao()
print("Custo da solução aleatória inicial: {custo}".format(custo = custo_total(solucao_inicial)))

print("Calculando rotas.....\n")
init_time = time.time()
solucao_final, cost = annealing(solucao_inicial)
end_time = time.time()
print("Solução Final \n {} \n".format(solucao_final))
print("Custo Final: ", cost)
print("Tempo total de execução: {time} \n".format(time = str(end_time - init_time)))

G = nw.Graph()
for i in range(len(solucao_final)):
    node = solucao_final[i]
    G.add_node(node , pos = (bairros[str(node)][0], bairros[str(node)][1]))
for i in range(len(solucao_final)):
    if i == len(solucao_final) - 1:
        G.add_edge(solucao_final[i], solucao_final[0])
    else :
        G.add_edge(solucao_final[i], solucao_final[i + 1])

pos = nw.get_node_attributes(G, 'pos')
nw.draw(G, pos, node_size = 10, with_labels=True)
plt.show()