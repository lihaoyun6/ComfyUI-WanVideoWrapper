import torch
import math
import numpy as np
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

class FlowMatchEulerDynamicScheduler(FlowMatchEulerDiscreteScheduler):
    """
    Планировщик Euler/d - на основе Euler, но с внедрением архитектуры предиктор-корректор
    
    Этот планировщик основан на методе Эйлера, но добавляет расширенные функции, заимствованные из стандартного сэмплера UniPC,
    включая предиктор второго порядка и методы коррекции.
    Он направлен на сохранение простоты Euler при значительном повышении скорости сэмплирования и качества образцов,
    особенно подходит для задач создания видео.
    """
    
    def __init__(self, shift=3.0, use_beta_sigmas=False):
        super().__init__(shift=shift, use_beta_sigmas=use_beta_sigmas)
        self.name = "euler_d"
        # Хранение исторических данных предыдущих шагов
        self._model_outputs = []
        self._samples = []
        self._timesteps = []
        # Максимальное количество сохраняемых записей
        self._max_history = 3
        # Коэффициент ускорения
        self._acceleration_factor = 1.5
    
    def _advanced_euler_step(self, model_output, sample, timestep_index, prev_timestep_index):
        """
        Улучшенный шаг Эйлера, использующий архитектуру предиктор-корректор
        """
        # Получение значений sigma
        sigma = self.sigmas[timestep_index]
        sigma_next = self.sigmas[prev_timestep_index] if prev_timestep_index < len(self.sigmas) else 0
        
        # Базовый размер шага - определяется сигмой
        step_size = sigma - sigma_next
        
        # Вычисление стандартного направления Эйлера
        direction = -model_output
        
        # Проверка наличия достаточного количества исторических данных для применения предиктора-корректора
        if len(self._model_outputs) >= 2:
            # Фаза предиктора - использование стандартного метода Эйлера для предсказания следующей точки
            predicted_sample = sample + step_size * direction
            
            # Использование исторических данных для расчета более точного направления
            # На основе скорости изменения между предыдущим и пред-предыдущим шагом
            prev_direction = -self._model_outputs[-1]
            prev_prev_direction = -self._model_outputs[-2]
            
            # Вычисление производных направления
            direction_derivative = (direction - prev_direction) / \
                                  (self._timesteps[-1] - timestep_index)
            direction_second_derivative = ((direction - prev_direction) - \
                                        (prev_direction - prev_prev_direction)) / \
                                       ((self._timesteps[-1] - timestep_index) * \
                                        (self._timesteps[-2] - self._timesteps[-1]))
            
            acceleration = 0.5 * step_size * step_size * direction_second_derivative
            velocity = step_size * (direction + 0.5 * step_size * direction_derivative)
            
            acceleration_weight = self._acceleration_factor * (1.0 - np.exp(-timestep_index / len(self.sigmas) * 5))
            
            corrected_sample = sample + velocity + acceleration_weight * acceleration
            
            correction_weight = min(0.8, timestep_index / len(self.sigmas) * 1.0)
            next_sample = (1.0 - correction_weight) * predicted_sample + correction_weight * corrected_sample
            
            critical_start = int(len(self.sigmas) * 0.05)
            critical_end = int(len(self.sigmas) * 0.8)
            
            if critical_start <= timestep_index <= critical_end:
                enhancement_factor = 1.2
                enhancement_weight = torch.sigmoid(torch.tensor((timestep_index - critical_start) / \
                                                          (critical_end - critical_start) * 10 - 5)).item()
                
                next_sample = (1.0 - enhancement_weight) * next_sample + \
                             enhancement_weight * (sample + enhancement_factor * step_size * direction)
        else:
            next_sample = sample + step_size * direction
        
        self._model_outputs.append(model_output.clone())
        self._samples.append(sample.clone())
        self._timesteps.append(timestep_index)
        
        if len(self._model_outputs) > self._max_history:
            self._model_outputs.pop(0)
            self._samples.pop(0)
            self._timesteps.pop(0)
        
        return next_sample
    
    def step(self, model_output, timestep, sample, **kwargs):
        step_index = (self.timesteps == timestep).nonzero().item()
        prev_step_index = step_index + 1 if step_index + 1 < len(self.sigmas) else step_index

        next_sample = self._advanced_euler_step(
            model_output=model_output,
            sample=sample,
            timestep_index=step_index,
            prev_timestep_index=prev_step_index
        )
        
        return next_sample
    
    def set_timesteps(self, num_inference_steps, device=None, sigmas=None, scheduler_type=None):
        try:
            if scheduler_type is not None:
                super().set_timesteps(num_inference_steps, device, sigmas, scheduler_type)
            else:
                super().set_timesteps(num_inference_steps, device, sigmas)
        except TypeError:
            super().set_timesteps(num_inference_steps, device, sigmas)
            
        self._model_outputs = []
        self._samples = []
        self._timesteps = []
        
        if num_inference_steps < 20:
            self._acceleration_factor = 2.0
        elif num_inference_steps < 30:
            self._acceleration_factor = 1.5
        else:
            self._acceleration_factor = 1.2
