#include <mpi.h>
#include <omp.h>

#include <numeric>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <time.h>
#include <sys/time.h>
#include <sstream>
#include <string>
#include <fstream>
#include "particle.h"
#undef SEEK_SET
#undef SEEK_CUR
#undef SEEK_END
const double ARmass = 39.948; // A.U.s
const double ARsigma = 3.405; // Angstroms
const double AReps = 119.8;	  // Kelvins
const double CellDim = 12.0;  // Angstroms
int NPartPerCell = 10;

using namespace std;

double WallTime(void){
	static long zsec = 0;
	static long zusec = 0;
	double esec;
	struct timeval tp;
	struct timezone tzp;
	gettimeofday(&tp, &tzp);
	if (zsec == 0)
		zsec = tp.tv_sec;
	if (zusec == 0)
		zusec = tp.tv_usec;
	esec = (tp.tv_sec - zsec) + (tp.tv_usec - zusec) * 0.000001;
	return esec;
}

double update(struct Particle &atom){
	// We are using a 1.0 fs timestep, this is converted
	double DT = 0.000911633;
	atom.ax = atom.fx / ARmass;
	atom.ay = atom.fy / ARmass;
	atom.vx += atom.ax * 0.5 * DT;
	atom.vy += atom.ay * 0.5 * DT;
	atom.x += DT * atom.vx;
	atom.y += DT * atom.vy;
	atom.fx = 0.0;
	atom.fy = 0.0;
	return 0.5 * ARmass * (atom.vx * atom.vx + atom.vy * atom.vy);
}

double interact(struct Particle &atom1, struct Particle &atom2){
	double rx, ry, rz, r, fx, fy, fz, f;
	double sigma6, sigma12;

	// computing base values
	rx = atom1.x - atom2.x;
	ry = atom1.y - atom2.y;
	r = rx * rx + ry * ry;

	if (r < 0.000001)
		return 0.0;

	r = sqrt(r);
	sigma6 = pow((ARsigma / r), 6);
	sigma12 = sigma6 * sigma6;
	f = ((sigma12 - 0.5 * sigma6) * 48.0 * AReps) / r;
	fx = f * rx;
	fy = f * ry;
	// updating particle properties
	atom1.fx += fx;
	atom1.fy += fy;
	return 4.0 * AReps * (sigma12 - sigma6);
}

int main(int argc, char *argv[]){
	if (argc != 3){
		cerr << "Incorrect syntax: should be two arguments";
		exit(2);
	}
	int rank,size;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	int NumParticles = atoi(argv[1]);
	int NumIterations = atoi(argv[2]);
	int Dimension = (int)(sqrt(NumParticles / (double)(NPartPerCell)));
	int NCellRows = Dimension;
	int NCellCols = Dimension;
	int TotalCells = Dimension * Dimension;
	struct Particle particles[NCellRows][NCellCols][NPartPerCell];

	int nrow_proc=Dimension/size;

	if(rank==0)
		cout << "\nThe Total Number of Cells is " << TotalCells
			 << " With a maximum of " << NPartPerCell << " particles per cell,"
			 << "\nand " << NumParticles << " particles total in system\n";

	const char *filename = "data.bin";

	ifstream infile(filename, ios::binary);
	infile.read((char *)particles, sizeof(particles));
	infile.close();

	int cx_start=rank*nrow_proc; //左闭右开
	int cx_end=(rank+1)*nrow_proc;
	struct Particle local_particles[nrow_proc+2][NCellCols][NPartPerCell];
	// copy to local particles
	for(int cx=1;cx<=nrow_proc;cx++){
		for(int cy=0;cy<NCellCols;cy++){
			for(int cz=0;cz<NPartPerCell;cz++){
				local_particles[cx][cy][cz]=particles[cx_start+cx-1][cy][cz];
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double TimeStart = WallTime();
	for(int t=0; t < NumIterations; t++){ // For each timestep
		double TotPot_local = 0.0;
		double TotKin_local = 0.0;
		int left_proc = rank>0?rank-1:MPI_PROC_NULL;
		int right_proc= rank<size-1?rank+1:MPI_PROC_NULL;
		MPI_Request send_to_left,recv_from_right,send_to_right,recv_from_left;
		//communication
		// send and recv the front row
		// MPI_Sendrecv(local_particles[1][0],NCellCols*NPartPerCell*8,MPI_DOUBLE,left_proc,1,
		// 				local_particles[nrow_proc+1][0],NCellCols*NPartPerCell*8,MPI_DOUBLE,right_proc,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		MPI_Isend(local_particles[1][0],NCellCols*NPartPerCell*8,MPI_DOUBLE,left_proc,1,MPI_COMM_WORLD,&send_to_left);
		MPI_Irecv(local_particles[nrow_proc+1][0],NCellCols*NPartPerCell*8,MPI_DOUBLE,right_proc,1,MPI_COMM_WORLD,&recv_from_right);
		// send and recv the last row
		// MPI_Sendrecv(local_particles[nrow_proc][0],NCellCols*NPartPerCell*8,MPI_DOUBLE,right_proc,2,
		// 				local_particles[0][0],NCellCols*NPartPerCell*8,MPI_DOUBLE,left_proc,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		MPI_Isend(local_particles[nrow_proc][0],NCellCols*NPartPerCell*8,MPI_DOUBLE,right_proc,2,MPI_COMM_WORLD,&send_to_right);
		MPI_Irecv(local_particles[0][0],NCellCols*NPartPerCell*8,MPI_DOUBLE,left_proc,2,MPI_COMM_WORLD,&recv_from_left);
		#pragma omp parallel for reduction(+:TotPot_local)
		for(int cx1 = 1; cx1 <= nrow_proc; cx1++){
			for(int cy1 = 0; cy1 < NCellCols; cy1++){
				//与自己的相互作用力
				for(int i=0;i<NPartPerCell;i++){
					for(int j=0;j<NPartPerCell;j++){
						if(i!=j)
							TotPot_local += interact(local_particles[cx1][cy1][i],local_particles[cx1][cy1][j]);
					}
				}
			}
		}
		//middle rows
		#pragma omp parallel for reduction(+:TotPot_local)
		for(int cx1=2;cx1<nrow_proc;cx1++){
			for(int cy1=0;cy1<NCellCols;cy1++){
				for(int cx2=1;cx2<nrow_proc+1;cx2++){
					for(int cy2=0;cy2<NCellCols;cy2++){
						if((abs(cx1-cx2)<2) && (abs(cy1-cy2)<2) && (cx1!=cx2 || cy1!=cy2)){
							for(int i=0;i<NPartPerCell;i++){
								// for(int j=0;j<NPartPerCell;j++){
								// 	TotPot_local += interact(local_particles[cx1][cy1][i],local_particles[cx2][cy2][j]);
								// }
								TotPot_local += interact(local_particles[cx1][cy1][i],local_particles[cx2][cy2][0])
												+interact(local_particles[cx1][cy1][i],local_particles[cx2][cy2][1])
												+interact(local_particles[cx1][cy1][i],local_particles[cx2][cy2][2])
												+interact(local_particles[cx1][cy1][i],local_particles[cx2][cy2][3])
												+interact(local_particles[cx1][cy1][i],local_particles[cx2][cy2][4])
												+interact(local_particles[cx1][cy1][i],local_particles[cx2][cy2][5])
												+interact(local_particles[cx1][cy1][i],local_particles[cx2][cy2][6])
												+interact(local_particles[cx1][cy1][i],local_particles[cx2][cy2][7])
												+interact(local_particles[cx1][cy1][i],local_particles[cx2][cy2][8])
												+interact(local_particles[cx1][cy1][i],local_particles[cx2][cy2][9]);

							}
						}
					}
				}
			}
		}
		MPI_Wait(&send_to_left,MPI_STATUS_IGNORE);
		MPI_Wait(&recv_from_right,MPI_STATUS_IGNORE);
		MPI_Wait(&send_to_right,MPI_STATUS_IGNORE);
		MPI_Wait(&recv_from_left,MPI_STATUS_IGNORE);
		//与其他cell的相互作用力
		// first row
		#pragma omp parallel for reduction(+:TotPot_local)
		for(int cy1=0;cy1<NCellCols;cy1++){
			int cx2_start=rank==0?1:0;
			int cx2_end=3;
			for(int cx2=cx2_start;cx2<cx2_end;cx2++){
				for(int cy2=0;cy2<NCellCols;cy2++){
					if((abs(1-cx2)<2) && (abs(cy1-cy2)<2) && (1!=cx2 || cy1!=cy2)){
						for(int i=0;i<NPartPerCell;i++){
							// for(int j=0;j<NPartPerCell;j++){
							// 	TotPot_local += interact(local_particles[1][cy1][i],local_particles[cx2][cy2][j]);
							// }
							double sub1=0.0,sub2=0.0;
							for(int j=0;j<NPartPerCell;j+=2){
								sub1+=interact(local_particles[1][cy1][i],local_particles[cx2][cy2][j]);
								sub2+=interact(local_particles[1][cy1][i],local_particles[cx2][cy2][j+1]);
							}
							TotPot_local += sub1+sub2;
						}

					}
				}
			}
		}
		// last row
		#pragma omp parallel for reduction(+:TotPot_local)
		for(int cy1=0;cy1<NCellCols;cy1++){
			int cx2_start=nrow_proc-1;
			int cx2_end=rank==size-1?nrow_proc+1:nrow_proc+2;
			for(int cx2=cx2_start;cx2<cx2_end;cx2++){
				for(int cy2=0;cy2<NCellCols;cy2++){
					if((abs(nrow_proc-cx2)<2) && (abs(cy1-cy2)<2) && (nrow_proc!=cx2 || cy1!=cy2)){
						for(int i=0;i<NPartPerCell;i++){
							for(int j=0;j<NPartPerCell;j++){
								TotPot_local += interact(local_particles[nrow_proc][cy1][i],local_particles[cx2][cy2][j]);
							}
						}
					}
				}
			}
		}


		// End iteration over cells
		// Apply the accumulated forces; update accelerations, velocities, positions
		#pragma omp parallel for reduction(+:TotKin_local)
		for(int cx1=1;cx1<=nrow_proc;cx1++){
			for(int cy1=0;cy1<NCellCols;cy1++){
				for(int i=0;i<NPartPerCell;i++){
					TotKin_local +=update(local_particles[cx1][cy1][i]);
				}
			}
		}
		double local_sum=TotKin_local+TotPot_local;
		double sum=0.0;
		MPI_Reduce(&local_sum,&sum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		if(rank==0){
			printf("\nIteration#%d with Total Energy %e per Atom", t, (sum) / NumParticles);
		}
	}
	double TimeEnd = WallTime();
	if(rank==0)
		cout << "\nTime for " << NumIterations << " is " << TimeEnd - TimeStart << endl;
	MPI_Finalize();
	return 0;
}
