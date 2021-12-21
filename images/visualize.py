import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def main():
    x = np.loadtxt("../results/input.txt")
    y = np.loadtxt("../results/output.txt")
    y_back = np.loadtxt("../results/output_back.txt")

    x_spec = np.loadtxt("../results/input_spectr.txt")
    y_spec = np.loadtxt("../results/output_spectr.txt")
    y_back_spec = np.loadtxt("../results/output_back_spectr.txt")

    t = np.loadtxt("../results/time.txt")
    w = np.loadtxt("../results/freq.txt")

    E_start = x[::2] + 1j*x[1::2]
    E_end = y[::2] + 1j * y[1::2]
    E_back = y_back[::2] + 1j * y_back[1::2]

    E_startS = x_spec[::2] + 1j*x_spec[1::2]
    E_endS = y_spec[::2] + 1j*y_spec[1::2]
    E_backS = y_back_spec[::2] + 1j*y_back_spec[1::2]

    I_start = np.abs(E_start*np.conj(E_start))
    I_end = np.abs(E_end * np.conj(E_end))
    I_back = np.abs(E_back * np.conj(E_back))

    I_startS = np.abs(E_startS*np.conj(E_startS))
    I_endS = np.abs(E_endS*np.conj(E_endS))
    I_backS = np.abs(E_backS*np.conj(E_backS))

    W_start = np.trapz(I_start, t)
    W_end = np.trapz(I_end, t)
    W_back = np.trapz(I_back, t)

    print("Energy input = %.2f" % W_start)
    print("Energy back = %.2f" % W_back)
    print("Energy output = %.2f" % W_end)

    # Input and back propagated signal intensity
    nt1 = abs(t - (-40)).argmin()
    nt2 = abs(t - 50).argmin()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[nt1:nt2], y=I_start[nt1:nt2], mode='lines', name='Input'))
    fig.add_trace(go.Scatter(x=t[nt1:nt2], y=I_back[nt1:nt2], mode='lines', name='Back'))
    fig.update_layout(title_text="Signal intensity")
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Intensity")
    fig.write_html("input.html")

    # Output signal intensity
    fig = px.line(x=t, y=I_end, title='Output signal intensity', labels={'x': 'Intensity', 'y': 'Time'})
    fig.write_html("output.html")

    # Spectrum
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w, y=I_startS, mode='lines', name='Input'))
    fig.add_trace(go.Scatter(x=w, y=I_endS, mode='lines', name='Output'))
    fig.add_trace(go.Scatter(x=w, y=I_backS, mode='lines', name='Back'))
    fig.update_layout(yaxis_type="log")
    fig.update_layout(title_text="Spectrum intensity")
    fig.update_xaxes(title_text="Frequency")
    fig.update_yaxes(title_text="Intensity (log scale)")
    fig.write_html("spectrum.html")


if __name__ == '__main__':
    main()
