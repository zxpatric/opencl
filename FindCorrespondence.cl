_kernel void FindMatch(__global __read_only float* apm, 
						__global __read_only uchar* vm,
						ushort rows,
						ushort cols,
						__global __read_only float8* crystles, 
						ushort rowStride0,
						ushort offset,
						__global __read_only unsigned short* roughMap,
						ushort rowStride1,
						__global __read_only unsigned short* fineMap,
						ushort rowStride2,
						float3 fmLR0,
						float3 fmLR1,
						float3 fmLR2,
						__global __write_only float2* cm,
						__global __write_only uchar*  ivm)
{
	int rowStride = cols;
	int row = get_global_id(1);
	int col = get_global_id(0);
	int idx = IDX(row, col);
	
	ivm[idx] = 1;
	float phase = apm[idx];
	if (vm[idx] && phase > 0.0)
	{
		float l0 = fmLR0.s0*col + fmLR0.s1*row + fmLR0.s2;
		float l1 = fmLR1.s0*col + fmLR1.s1*row + fmLR1.s2;
		float l2 = fmLR2.s0*col + fmLR2.s1*row + fmLR2.s2;

		unsigned short colBoundary = cols - 2;
		unsigned short rowBoundary = rows - 2;

		unsigned short cL = 0;
		float frL = - l2 / l1;

		unsigned short cR = colBoundary; //cols - 1;		// maxPhase
		float frR = -((float)(cR)*l0 + l2) / l1;

		unsigned short lr, lc;
		if (frL < 0)
		{
			lr = 0;
			lc = (unsigned short)(-l2 / l0);
		}
		else if (frL > rowBoundary)
		{
			lr = rowBoundary;
			lc = (unsigned short)((-l2-l1*(float)rowBoundary) / l0);
		}
		else
		{
			lr = (unsigned short)frL;
			lc = 0;
		}


		unsigned short rr, rc;
		if (frR < 0)
		{
			rr = 0;
			rc = (unsigned short)(-l2 / l0);
		}
		else if (frR > rowBoundary)
		{
			rr = rowBoundary;
			rc = (unsigned short)(-(l2 + l1*(float)rowBoundary) / l0);
		}
		else
		{
			rr = (unsigned short)frR;
			rc = colBoundary;
		}
		unsigned short rMin = min(lr, rr);
		unsigned short rMax = max(lr, rr);
		unsigned short cMin = min(lc, rc);
		unsigned short cMax = max(lc, rc);

		float frId = frL;
		unsigned short roughSeed = roughMap[IDX2((unsigned short)(frId + offset + 0.5f), rowStride1, (unsigned short)(phase + 0.5f))];
		unsigned short middleSeed = fineMap[IDX2((unsigned short)((frId + offset)*2.0f + 0.5f), rowStride2, (unsigned short)(phase*2.0f + 0.5f))];

		unsigned short r, c;
		float fr;
		if (middleSeed > 0)
		{
			c = middleSeed;
			fr = -(l2 + l0 * middleSeed) / l1;

			frL = select(fr-4.0f, 0.0f, fr<4.0f);
			//rMin = max(rMin, (fr<4.0f) ? (unsigned short)0 : (unsigned short)(fr-4.0f));
			rMin = max(rMin, (unsigned short)frL);
			rMax = min(rMax, (unsigned short)(rMin + 9));
			//cMin = max(cMin, (middleSeed<4) ? (unsigned short)0 : (unsigned short)(middleSeed - 4));
			r = select(middleSeed - 4, 0, middleSeed<4);
			cMin = max(cMin, r);
			cMax = min(cMax, (unsigned short)(cMin + 9));
		}
		else if (roughSeed > 0)
		{
			c = roughSeed;
			fr = -(l2 + l0 * roughSeed) / l1;

			// search around fr and fineSeed for the crystle.
			frL = select(fr-8.0f, 0.0f, fr<8.0f);
			rMin = max(rMin, (unsigned short)frL);
			rMax = min(rMax, (unsigned short)(rMin + 17));
			r = select(roughSeed - 8, 0, roughSeed<8);
			cMin = max(cMin, r);
			cMax = min(cMax, (unsigned short)(cMin + 17));
		}
		else
		{
			// ask if it's neighbor has any seed.
			r = (unsigned short)(frId + offset + 0.5f);
			unsigned short p = (unsigned short)(phase + 0.5f);

			if ((c=roughMap[IDX2(r, rowStride1, p-1)]) > 0)
			{
				fr = -(l2 + l0 * c) / l1;
			}
			else if ((c = roughMap[IDX2(r, rowStride1, p+1)]) > 0)
			{
				fr = -(l2 + l0 * c) / l1;
			}
			else if ((c = roughMap[IDX2(r-1, rowStride1, p)]) > 0)
			{
				fr = -(l2 + l0 * c) / l1;
			}
			else if ((c = roughMap[IDX2(r + 1, rowStride1, p)]) > 0)
			{
				fr = -(l2 + l0 * c) / l1;
			}
			else
			{
				//c = (cMin + cMax) / 2;
				//fr = -(l2 + l0 * c) / l1;
				return;
			}

			if (fr < rMin || fr > rMax)
			{
				return;
			}
		}

		r = (unsigned short)fr;
		unsigned short lastLRow, lastLCol, lastRRow, lastRCol;

		int idxc = IDX2(r, rowStride0, c);
		float8 co = crystles[idxc];
		if (co.s3 > 0 && co.s2 <= phase && phase <= co.s3 && co.s1 <= frId && frId <= co.s2)
		{

		}
		else
		{
			if (co.s3 > 0)
			{
				lastRRow = lastLRow = r;
				lastRCol = lastLCol = c;
			}
			else
			{
				lastLRow=0;
				lastLCol=0;
				lastRRow=0;
				lastRCol=0;
			}
	
			unsigned short step = 1;
			unsigned short jl = c - step;
			unsigned short jr = c + step;
			bool checkLeft = true;
			bool checkRight = true;
			while (checkLeft || checkRight)
			{
				if (jl <= 0)
				{
					checkLeft = false;
				}

				if (jr > cMax)
				{
					checkRight = false;
				}

				if (checkLeft)
				{
					fr = -(l2 + l0 * jl) / l1;
					unsigned short rr = (unsigned short)fr;
					if (rr<rMin || rr>rMax)
					{
						checkLeft = false;
					}
					else
					{
						idxc = IDX2(rr, rowStride0, jl);
						co = crystles[idxc];
						if (co.s3 > 0)
						{
							if (co.s2 <= phase && phase <= co.s3 && co.s1 <= frId && frId <= co.s2)
							{
								r = rr;
								c = jl;
								break;
							}
							else
							{
								if (lastLRow + lastLCol)
								{
									//checkLeft = isCloser(rr, jl, frId, phase, lastLRow, lastLCol);
									int idxlast = IDX2(rr, rowStride0, jl);
									float8 last_co = crystles[idxlast];
									if (phase < co.s2 && phase < last_co.s2)
									{
										checkLeft = co.s2 < last_co.s2;
									}

									if (phase > co.s3 && phase > last_co.s3)
									{
										checkLeft = co.s3 > last_co.s3;
									}
								}

								lastLRow = rr;
								lastLCol = jl;
							}
						}
						jl -= step;
					}
				}

				if (checkRight)
				{
					fr = -(l2 + l0 * jr) / l1;
					unsigned short rr = (unsigned short)fr;

					if (rr<rMin || rr>rMax)
					{
						checkRight = false;
					}
					else
					{
						idxc = IDX2(rr, rowStride0, jr);
						co = crystles[idxc];

						if (co.s3 > 0)
						{
							if (co.s2 <= phase && phase <= co.s3 && co.s1 <= frId && frId <= co.s2)
							{
								r = rr;
								c = jr;
								break;
							}
							else
							{
								if (lastRRow + lastRCol)
								{
									//checkRight = isCloser(rr, jr, frId, phase, lastRRow, lastRCol);
									int idxlast = IDX2(rr, rowStride0, jr);
									float8 last_co = crystles[idxlast];
									if (phase < co.s2 && phase < last_co.s2)
									{
										checkRight = co.s2 < last_co.s2;
									}

									if (phase > co.s3 && phase > last_co.s3)
									{
										checkRight = co.s3 > last_co.s3;
									}
								}
							}
							lastRRow = rr;
							lastRCol = jr;
						}
						jr += step;
					}
				}
			}
		}

		unsigned short r1 = r;
		unsigned short c1 = c;
		unsigned short r2 = r1 + 1;
		unsigned short c2 = c1 + 1;

		co = crystles[IDX2(r1, rowStride0, c1)];
		float LU = co.s4;
		float LD = co.s5;
		float RD = co.s6;
		float RU = co.s7;
		// search in the (r, c) ~ (r+1, c+1) for phase

		float fc = c;
		float fr1 = r1;
		float fc1 = c1;
		float fr2 = r2;
		float fc2 = c2;

		float change = 1.0f;
		float e = 1000.0f;
		float e_last;
		float CurrentPhsC, fCurrentPhsC, JaconbinInverse, c0;
		int iter = 0;
		float cColChangeThresh = 0.0001f;
		float cPhaseDiffThresh = 0.0001f;
		int MaxIter = 5;

		float e01 = l0 / l1;
		float e21 = l2 / l1;
		float e01double = e01 * 2.0f;

		while (++iter < MaxIter)
		{
			fr = -(l2 + l0 * fc) / l1;

			CurrentPhsC = (fc2 - fc) * (fr2 - fr) * LU
				+ (fc - fc1) * (fr2 - fr) * RU
				+ (fc2 - fc) * (fr - fr1) * LD
				+ (fc - fc1) * (fr - fr1) * RD;
			e_last = e;
			e = phase - CurrentPhsC;
			if ((fabs(e_last) - fabs(e) < cPhaseDiffThresh) || (fabs(e) < cPhaseDiffThresh) || (fabs(change) < cColChangeThresh))
			{
				break;
			}

			fCurrentPhsC = (-e01double * fc - e21 - fr2 + fc2*e01)*LU
				+ (-e01double * fc - e21 - fr1 + fc1*e01)*RD
				+ (e01double * fc + e21 + fr2 - fc1*e01)*RU
				+ (e01double * fc + e21 + fr1 - fc2*e01)*LD;

			c0 = fabs(fCurrentPhsC);
			if (c0 < FLT_EPSILON || c0 > 1.0)
			{
				return;
			}

			JaconbinInverse = 1 / fCurrentPhsC;
			change = JaconbinInverse*e;
			fc = fc + change;
		}

		cm[idx].s0 = fc;
		cm[idx].s1 = fr;
		ivm[idx] = 0;
	}	
}