import numpy as np
def generate_walsh_matrix(n):
    """Generate Walsh code matrix of size 2^n x 2^n"""
    if n == 1:
        return np.array([[1, 1], [1, -1]])
    smaller_matrix = generate_walsh_matrix(n - 1)
    return np.vstack((
    np.hstack((smaller_matrix, smaller_matrix)),
    np.hstack((smaller_matrix, -smaller_matrix))
    ))

def cdma_simulation(num_users=8, intended_bit=1, noise_floor=1e-6):
    """Run CDMA simulation with detailed power analysis"""
    # Generate and display Walsh matrix
    walsh_matrix = generate_walsh_matrix(int(np.log2(num_users)))
    print("Walsh Code Matrix:")
    print(walsh_matrix)
    # print("\n" + "="*50 + "\n")
    # Select intended user
    intended_index = np.random.randint(0, num_users)
    print(f"Intended User: {intended_index+1}, Transmitting bit: {intended_bit}\n")
    # Create transmitted signal (convert to float for noise addition)
    transmitted_signal = intended_bit * walsh_matrix[intended_index].astype(np.float64)
    # Add Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_floor), num_users)
    noisy_signal = transmitted_signal + noise

    print("Transmitted Signal (before noise):")
    print(transmitted_signal)
    print("\nNoise Added:")
    print(noise)
    print("\nReceived Signal (with noise):")
    print(noisy_signal)
    print("\n" + "="*50 + "\n")

    # Calculate metrics for all users
    results = []
    for i in range(num_users):
        # Correlation detection
        correlation = np.dot(noisy_signal, walsh_matrix[i]) / num_users
        # Signal power (desired component)
        signal_power = (intended_bit * (i == intended_index)) ** 2
        # Noise power (interference + channel noise)
        if i == intended_index:
        # For intended user: noise comes from channel only
            noise_power = np.var(noise)
        else:
        # For unintended users: noise comes from intended signal + channel noise
            interference = intended_bit * np.dot(walsh_matrix[intended_index], walsh_matrix[i]) / num_users
            noise_power = interference**2 + np.var(noise)
        # Calculate SNR (handle case where noise_power is 0)
        snr = signal_power / noise_power if noise_power > 0 else float('inf')
        snr_db = 10 * np.log10(snr) if not np.isinf(snr) else float('inf')

        results.append({
        'user': i+1,
        'type': 'Intended' if i == intended_index else 'Unintended',
        'correlation': correlation,
        'signal_power': signal_power,
        'noise_power': noise_power,
        'snr_db': snr_db
        })

    print("User\tType\t\tCorrelation\tSignal Power\tNoise Power\tSNR (dB)")
    print("-"*80)
    for res in results:
        print(f"{res['user']}\t{res['type']:12}\t{res['correlation']:+.4f}\t"f"{res['signal_power']:.6f}\t{res['noise_power']:.6f}\t"f"{res['snr_db']:.2f}")
    # Summary statistics
    intended = [r for r in results if r['type'] == 'Intended'][0]
    unintended = [r for r in results if r['type'] == 'Unintended']
    avg_unintended_snr = np.mean([r['snr_db'] for r in unintended])

    print("\n" + "="*50)
    print("SUMMARY RESULTS:")
    print(f"Intended User {intended['user']}:")
    print(f" - Correlation: {intended['correlation']:+.4f}")
    print(f" - SNR: {intended['snr_db']:.2f} dB")
    print(f" - Signal Power: {intended['signal_power']:.6f}")
    print(f" - Noise Power: {intended['noise_power']:.6f}")
    print(f"\nAverage Unintended User:")
    print(f" - Avg Correlation: {np.mean([r['correlation'] for r in unintended]):+.6f}")
    print(f" - Avg SNR: {avg_unintended_snr:.2f} dB")
    print(f" - Avg Signal Power: {np.mean([r['signal_power'] for r in unintended]):.6f}")
    print(f" - Avg Noise Power: {np.mean([r['noise_power'] for r in unintended]):.6f}")
    print("="*50)
# Run simulation
cdma_simulation(num_users=8, intended_bit=1, noise_floor=1e-6)




''' Walsh Code Matrix:
// [[ 1 1 1 1 1 1 1 1]
// [ 1 -1 1 -1 1 -1 1 -1]
// [ 1 1 -1 -1 1 1 -1 -1]
// [ 1 -1 -1 1 1 -1 -1 1]
// [ 1 1 1 1 -1 -1 -1 -1]
// [ 1 -1 1 -1 -1 1 -1 1]
// [ 1 1 -1 -1 -1 -1 1 1]
// [ 1 -1 -1 1 -1 1 1 -1]]
// ==================================================
// Intended User: 5, Transmitting bit: 1
// Transmitted Signal (before noise):
// [ 1. 1. 1. 1. -1. -1. -1. -1.]
// Noise Added:
// [ 8.44143595e-04 -1.09568784e-03 -7.56435802e-04 -1.57004781e-03
// 7.16470165e-04 -6.58332140e-04 -7.88668207e-04 -3.12023519e-05]
// Received Signal (with noise):
// [ 1.00084414 0.99890431 0.99924356 0.99842995 -0.99928353 -1.00065833
// -1.00078867 -1.0000312 ]
// ==================================================
// User Type Correlation Signal Power Noise Power SNR (dB)
// --------------------------------------------------------------------------------
// 1 Unintended -0.0004 0.000000 0.000001 -inf
// 2 Unintended +0.0004 0.000000 0.000001 -inf
// 3 Unintended +0.0004 0.000000 0.000001 -inf
// 4 Unintended +0.0004 0.000000 0.000001 -inf
// 5 Intended +0.9998 1.000000 0.000001 61.93
// 6 Unintended +0.0003 0.000000 0.000001 -inf
// 7 Unintended +0.0001 0.000000 0.000001 -inf
// 8 Unintended -0.0001 0.000000 0.000001 -inf
// ==================================================
// SUMMARY RESULTS:
// Intended User 5:
// - Correlation: +0.9998
// - SNR: 61.93 dB
// - Signal Power: 1.000000
// - Noise Power: 0.000001
// Average Unintended User:
// - Avg Correlation: +0.000153
// - Avg SNR: -inf dB
// - Avg Signal Power: 0.000000
// - Avg Noise Power: 0.000001
// ==================================================
// C:\Users\My\AppData\Local\Temp\ipykernel_25616\2023421112.py:60: RuntimeWarning: divide by zero encountered in log10
// snr_db = 10 * np.log10(snr) if not np.isinf(snr) else float('inf')
// print("\n" + "="*50)
// print("SUMMARY RESULTS:")
// print(f"Intended User {intended['user']}:")
// print(f" - Correlation: {intended['correlation']:+.4f}")
// print(f" - SNR: {intended['snr_db']:.2f} dB")
// print(f" - Signal Power: {intended['signal_power']:.6f}")
// print(f" - Noise Power: {intended['noise_power']:.6f}")
'''