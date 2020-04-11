import numpy as np
import qutip as qt
import sys


# implementazione hamiltoniana

class Hamiltonian(qt.Qobj):
    def __init__(self, filename=None, mat=None, calculate_eigens=False):
        if filename is None and mat is None:
            sys.exit("Error creating hamiltonian")
        elif filename is None:
            self.mat = mat
        else:
            self.mat = np.loadtxt(filename)

        super().__init__(self.mat)
        self.noise = 0
        if calculate_eigens:
            self.eigenvalues, self.eigenkets = self.eigenstates()
        else:
            self.eigenvalues = []
            self.eigenkets = qt.Qobj()

    def calculate_eigensystem(self):
        self.eigenvalues, self.eigenkets = self.eigenstates()

    def get_eigenvalues(self):
        return self.eigenvalues

    def get_eigenkets(self):
        return self.eigenkets

    def matrix(self):
        return self.mat

    def dimension(self):
        return len(self.mat[0])

    def add_noise(self, noise):
        self.noise = noise
        # genera un vettore di numeri casuali uniformi tra -noise e +noise
        noise_vec = -noise + 2 * noise * np.random.random(self.dimension())
        self.mat = self.mat + np.identity(self.dimension()) * noise_vec
        new_ham = qt.Qobj(self.mat)
        self.__dict__.update(new_ham.__dict__)


class DeltaHamiltonian(Hamiltonian):
    def __init__(self, N, delta, is_cycle=False, calculate_eigens=False):
        mat = np.zeros(shape=(N, N))

        for i in range(N):
            if i != int((N - 1) / 2):
                mat[i, i] = 2
            else:
                mat[i, i] = 2 + delta
            if i != 0:
                mat[i - 1, i] = -1
                mat[i, i - 1] = -1

        if is_cycle:
            mat[0, N - 1] = -1
            mat[N - 1, 0] = -1

        super().__init__(mat=mat, calculate_eigens=calculate_eigens)


class Walker(qt.Qobj):
    def __init__(self, ham, pos_in):
        self.t = 0
        self.ham = ham
        self.pos_in = pos_in
        self.nsites = len(self.ham.matrix()[0])
        self.var = 0
        self.iscycle = False
        # se è cycle l'elemento in alto a destra è +- 1 (a seconda della convenzione)
        if abs(ham.matrix()[0][self.nsites - 1]) == 1:
            self.iscycle = True
        self.__dict__.update(qt.basis(self.nsites, pos_in).__dict__)
        self.initial_state = qt.basis(self.nsites, pos_in)

    def set_initial_state(self, state):
        self.initial_state = state

    def probabilities(self):
        return np.square(abs(self.full()[:, 0]))

    def ipr(self):
        prob = self.probabilities()
        # print(prob)
        sum_appo = 0
        for i in range(len(prob)):
            sum_appo = sum_appo + prob[i] * prob[i]

        return 1 / sum_appo

    def evolve(self, time):
        # self.__dict__.update(((-1j * (time - self.t) * self.ham).expm() * self).__dict__)
        # faccio l'evoluzione direttamente dallo stato iniziale, la riga commentata prima invece predispone all'evoluzione
        # fatta a partire da un generico stato (nel caso ce ne fosse bisogno)
        # self.__dict__.update(((-1j * time * self.ham).expm() * qt.basis(self.nsites, self.pos_in)).__dict__)
        self.__dict__.update(((-1j * time * self.ham).expm() * self.initial_state).__dict__)
        self.t = time

    # Da controllare la varianza: caso cycle
    def variance(self):
        media = float(0)
        mediaquad = float(0)
        distrib = self.probabilities()

        if self.iscycle:
            if self.nsites % 2 != 0:
                half = int((self.nsites - 1) / 2)
                # ne conto uno a destra e uno a sinistra, così i è la distanza da posizione iniziale
                # equivale (?) a mettere il sito iniziale in posizione zero
                # così gli altri vanno da -(N-1)/2 a (N-1)/2
                for i in range(half + 1):  # +1 per considerare siti +/- (N-1)/2
                    media += i * distrib[(self.pos_in + i) % self.nsites]
                    media += (-i) * distrib[(self.pos_in - i) % self.nsites]
                    mediaquad += i * i * distrib[(self.pos_in + i) % self.nsites]
                    mediaquad += i * i * distrib[(self.pos_in - i) % self.nsites]
            else:  # questo è ancora da testare
                half = int(self.nsites / 2)
                # ne conto uno a destra e uno a sinistra, così i è la distanza da posizione iniziale
                # equivale a mettere il sito iniziale in posizione zero
                # così gli altri vanno da -N/2+1 a +N/2

                for i in range(half):
                    media += i * distrib[(self.pos_in + i) % self.nsites]
                    media += (-i) * distrib[(self.pos_in - i) % self.nsites]
                    mediaquad += i * i * distrib[(self.pos_in + i) % self.nsites]
                    mediaquad += i * i * distrib[(self.pos_in - i) % self.nsites]

                # il ciclo considera i siti da -N/2+1 a +N/2-1, devo considerare anche il sito N/2
                media += half * distrib[(self.pos_in + half) % self.nsites]
                mediaquad += half * half * distrib[(self.pos_in + half) % self.nsites]

            # ALTERNATIVA: su un cycle d(j,k)=min(|j-k|, N-|j-k|)
            # |j-k| distanza in senso orario, N-|j-k| in senso antiorario
            # bisogna comunque il segno per calcolare la media

        else:
            for i in range(self.nsites):
                media += i * distrib[i]
                mediaquad += i * i * distrib[i]

        self.var = mediaquad - media * media
        return self.var

    def coherence(self):
        coherence = float(0)
        # self.full()[:,0] è vettore complesso, coefficienti che descrivono lo stato
        for j in self.full()[:, 0]:
            for k in self.full()[:, 0]:
                if j != k:
                    coherence += abs(np.conj(j) * k)
        return coherence

    def sum_probability(self, site_min, site_max):
        prob_sum = 0
        probabilities = self.probabilities()
        for i in range(site_max - site_min + 1):
            prob_sum += probabilities[site_min + i]
        return prob_sum


class ProbabilityDistribution():
    def __init__(self, p_in=None, dimensions=None):
        if p_in is None and dimensions is None:
            sys.exit("Probability distribution needs dimensions")
        elif p_in is None:
            self.prob_avg = np.zeros(dimensions)
            self.n_data = 0
            self.dimensions = dimensions
        else:
            self.prob_avg = p_in
            self.n_data = 1
            self.dimensions = len(p_in)

    def add_and_average(self, p_add):
        self.prob_avg = self.prob_avg * (self.n_data / (self.n_data + 1)) + p_add * (1 / (self.n_data + 1))
        # print(self.prob_avg)
        self.n_data += 1

    def get_average_probability(self):
        return self.prob_avg

    def dimension(self):
        return self.dimensions

    def print_txt(self, filename):
        file_out = open(filename, "w")
        output = ""
        for i in range(len(self.prob_avg)):
            output += str(self.prob_avg[i]) + "\n"

        file_out.write(output)


class GaussianWalker(Walker):
    def __init__(self, ham, mean, sigma, k0):
        self.t = 0
        self.ham = ham
        self.pos_in = mean
        self.sigma = sigma
        self.k0 = k0
        self.nsites = len(self.ham.matrix()[0])
        self.var = 0
        self.iscycle = False
        # se è cycle l'elemento in alto a destra è +- 1 (a seconda della convenzione)
        if abs(ham.matrix()[0][self.nsites - 1]) == 1:
            self.iscycle = True
        self.initial_coefficients = np.zeros(self.nsites, dtype=complex)
        for i in range(self.nsites):
            self.initial_coefficients[i] = np.exp(
                -0.5 * (self.pos_in - (i + 1)) ** 2 / self.sigma ** 2 + 1j * self.k0 * (i + 1))

        self.initial_state = qt.Qobj(self.initial_coefficients).unit()
        self.__dict__.update(self.initial_state.__dict__)


def calculate_fidelity(ket1, ket2):
    return np.abs((ket1.dag() * ket2).full()[0, 0])


def qfi(ket1, ket2, epsilon):
    return 8 * (1 - calculate_fidelity(ket1, ket2)) / epsilon ** 2
