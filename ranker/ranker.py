import os
import random
import pandas as pd
import tkinter as tk
from tkinter import messagebox, simpledialog
import pygame
from datetime import datetime
from compute_elo import Elo

class Ranker:
    def __init__(self, name):
        self.name = name or "default"
        self.elo = Elo()
        self.df = pd.read_parquet("./top_100.parquet")
        self.df["rating"] = self.df["rating"].astype(float)

        log_path = f"./rank_log_{self.name}.parquet"
        if os.path.exists(log_path):
            self.log = pd.read_parquet(log_path)
            self._apply_session_log()
        else:
            self.log = pd.DataFrame(columns=["id_1","id_2","winner","time"])

        self.master = tk.Tk()
        self.master.title(f"IrishMAN Ranker — User: {self.name}")
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.frame = tk.Frame(self.master, padx=20, pady=20)
        self.frame.pack()

        pygame.mixer.init()

        # 6) State for the current pair
        self.current_samples = None

    def _apply_session_log(self):
        for _, row in self.log.iterrows():
            samples = (int(row["id_1"]), int(row["id_2"]))
            winner  = int(row["winner"])
            self.update(samples, winner)

    def select(self):
        """Randomly pick two distinct indices from the DataFrame."""
        return random.sample(range(len(self.df)), 2)

    def play_song(self, song_id):
        """Load & play the corresponding MIDI file from irishman_midi/<id>.mid."""
        path = os.path.join("irishman_midi", f"{song_id}.mid")
        if not os.path.exists(path):
            messagebox.showerror("File Not Found", f"Could not find: {path}")
            return
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
        except Exception as e:
            messagebox.showerror("Playback Error", str(e))

    def update(self, samples, winner):
        """Apply Elo update in‐memory to self.df."""
        i1, i2 = samples
        id1 = self.df.iloc[i1]["id"]
        id2 = self.df.iloc[i2]["id"]
        r1  = self.df.loc[self.df["id"]==id1, "rating"].iloc[0]
        r2  = self.df.loc[self.df["id"]==id2, "rating"].iloc[0]
        new1, new2 = self.elo.rank(r1, r2, winner)
        self.df.loc[self.df["id"]==id1, "rating"] = new1
        self.df.loc[self.df["id"]==id2, "rating"] = new2

    def update_log(self, samples, winner):
        """Append one comparison to the user’s log."""
        i1, i2 = samples
        self.log.loc[len(self.log)] = {
            "id_1":  i1,
            "id_2":  i2,
            "winner": winner,
            "time":  datetime.now()
        }

    def next_round(self):
        """Clear the frame, pick a new pair, and show them."""
        for w in self.frame.winfo_children():
            w.destroy()
        self.current_samples = self.select()
        i1, i2 = self.current_samples
        s1 = self.df.iloc[i1]["id"]
        s2 = self.df.iloc[i2]["id"]
        self.display_songs([s1, s2])

    def display_songs(self, songs):
        """Build the buttons & labels for the current pair."""
        # Song 1
        tk.Label(self.frame, text=f"Song 1: {songs[0]}").pack(pady=5)
        tk.Button(self.frame,
                  text="► Play Song 1",
                  command=lambda: self.play_song(songs[0])
                 ).pack(pady=5)

        # Song 2
        tk.Label(self.frame, text=f"Song 2: {songs[1]}").pack(pady=5)
        tk.Button(self.frame,
                  text="► Play Song 2",
                  command=lambda: self.play_song(songs[1])
                 ).pack(pady=5)

        # Winner choices
        tk.Button(self.frame,
                  text="🎵 Song 1 is better",
                  command=lambda: self.handle_winner(0)
                 ).pack(pady=5)
        tk.Button(self.frame,
                  text="🎵 Song 2 is better",
                  command=lambda: self.handle_winner(1)
                 ).pack(pady=5)

        # Manual save to main DF
        tk.Button(self.frame,
                  text="Save Rankings to Main",
                  command=self.save_df
                 ).pack(pady=10)

        # Quit
        tk.Button(self.frame,
                  text="Quit",
                  command=self.on_closing
                 ).pack(pady=5)

    def handle_winner(self, winner):
        """User clicked which was better → update Elo & log, then next."""
        self.update(self.current_samples, winner)
        self.update_log(self.current_samples, winner)
        self.next_round()

    def save_log(self):
        """Write the user-specific log to disk."""
        path = f"./rank_log_{self.name}.parquet"
        self.log.to_parquet(path)
        messagebox.showinfo("Log Saved", f"User log written to:\n{path}")

    def save_df(self):
        """Overwrite the main top_100.parquet with updated ratings, then clear this user’s session log."""
        # 1) overwrite main ratings
        path = "./top_100.parquet"
        self.df.to_parquet(path)
        messagebox.showinfo("Rankings Saved", f"Main DF updated:\n{path}")

        # 2) delete the session log file if it exists
        session_path = f"./rank_log_{self.name}.parquet"
        if os.path.exists(session_path):
            os.remove(session_path)

        # 3) reset the in-memory log
        self.log = pd.DataFrame(columns=["id_1", "id_2", "winner", "time"])
        messagebox.showinfo("Session Reset", "Your session log has been cleared.")


    def on_closing(self):
        """Prompt to save the log, then exit."""
        if messagebox.askyesno("Quit", "Save your session log before quitting?"):
            self.save_log()
        self.master.destroy()

    def run(self):
        self.next_round()
        self.master.mainloop()


if __name__ == "__main__":
    # Ask for user name first
    root = tk.Tk()
    root.withdraw()
    user = simpledialog.askstring("Your Name", "Enter your name (for personal log):")
    root.destroy()

    # Launch the ranker
    app = Ranker(name=user)
    app.run()
