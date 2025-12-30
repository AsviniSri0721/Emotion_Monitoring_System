import api from '../services/api';

export interface InterventionVideo {
    id: string;
    title: string;
    description: string;
    file_path: string;
    duration: number; // seconds
    created_at: string;
}

export const interventionVideosApi = {
    getAll: async () => {
        const response = await api.get('/interventions/videos');
        return response.data;
    },

    getOne: async (id: string) => {
        const response = await api.get(`/interventions/videos/${id}`);
        return response.data;
    },

    upload: async (formData: FormData) => {
        const response = await api.post('/interventions/videos/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },
};
